"""Text completions and SSE routes adapted from tinyserve's build_app."""
import asyncio
import json
import time
import uuid
from contextlib import aclosing, asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import GenerationConfig

from tiny_vllm.async_engine import AsyncEngine
from tiny_vllm.kv_cache import SamplingParams


class SamplingBody(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str | None = None
    temperature: Annotated[float, Field(ge=0)] | None = None
    top_p: Annotated[float, Field(gt=0, le=1)] | None = None
    top_k: Annotated[int, Field(strict=True)] | None = None
    max_tokens: Annotated[int, Field(strict=True, ge=1)] | None = None
    max_completion_tokens: Annotated[int, Field(strict=True, ge=1)] | None = None
    stream: Annotated[bool, Field(strict=True)] = False
    ignore_eos: Annotated[bool, Field(strict=True)] = False
    n: Literal[1] = 1
    stream_options: dict[Literal['include_usage'], bool] | None = None


class CompletionBody(SamplingBody):
    prompt: str | list[Annotated[int, Field(strict=True)]]


class TextPart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['text']
    text: str


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'user', 'assistant']
    content: str | list[TextPart]


class ChatBody(SamplingBody):
    messages: Annotated[list[Message], Field(min_length=1)]
    chat_template_kwargs: dict | None = None


def error(message, status=400, kind='invalid_request_error'):
    return JSONResponse({'error': {'message': message, 'type': kind}}, status_code=status)


def build_app(engine: AsyncEngine, model_name: str):
    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(engine.close)

    app = FastAPI(lifespan=lifespan)
    eng = engine.engine
    app.state.engine = engine
    defaults = {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1}
    try:
        shipped = GenerationConfig.from_pretrained(eng.cfg.model).to_diff_dict()
        defaults.update({key: shipped[key] for key in defaults if shipped.get(key) is not None})
    except (OSError, ValueError):
        pass

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        first = exc.errors()[0]
        location = '.'.join(str(part) for part in first['loc'] if part != 'body')
        return error(f'{location}: {first["msg"]}')

    def params_from(body, prompt_len):
        room = eng.cfg.max_model_len - prompt_len
        limit = body.max_completion_tokens or body.max_tokens or room
        values = {key: getattr(body, key) if getattr(body, key) is not None else defaults[key]
                  for key in defaults}
        return SamplingParams(**values, max_tokens=min(limit, room), ignore_eos=body.ignore_eos)

    async def respond(prompt, body, chat, request):
        if body.model is not None and body.model != model_name:
            return error('Requested model is not loaded', 404)
        if not engine.healthy:
            return error('Generation worker is unavailable', 503, 'server_error')
        ids = list(prompt) if isinstance(prompt, list) else eng.tokenizer.encode(prompt)
        if not 0 < len(ids) < eng.cfg.max_model_len:
            return error(f'Prompt must contain 1 to {eng.cfg.max_model_len - 1} tokens')
        if any(not 0 <= token < eng.runner.model.cfg.vocab_size for token in ids):
            return error('Prompt contains an invalid token ID')
        params = params_from(body, len(ids))
        created = int(time.time())
        rid = ('chatcmpl-' if chat else 'cmpl-') + uuid.uuid4().hex
        obj = 'chat.completion' if chat else 'text_completion'

        def choice(output, text):
            if chat:
                key = 'delta' if body.stream else 'message'
                content = {'content': text} if body.stream else {'role': 'assistant', 'content': text}
                return {'index': 0, key: content, 'finish_reason': output.finish_reason}
            return {'index': 0, 'text': text, 'finish_reason': output.finish_reason}

        def usage(output):
            return dict(prompt_tokens=output.num_prompt_tokens, completion_tokens=len(output.token_ids),
                        total_tokens=output.num_prompt_tokens + len(output.token_ids),
                        prompt_tokens_details={'cached_tokens': output.num_cached_tokens})

        def envelope(choices):
            return dict(id=rid, object=obj + ('.chunk' if chat and body.stream else ''),
                        created=created, model=model_name, choices=choices)

        if body.stream:
            async def events():
                final = None
                try:
                    async with aclosing(engine.generate(ids, params)) as outputs:
                        if chat:
                            first = envelope([{'index': 0, 'delta': {'role': 'assistant', 'content': ''},
                                               'finish_reason': None}])
                            yield f'data: {json.dumps(first)}\n\n'
                        async for output in outputs:
                            final = output
                            yield f'data: {json.dumps(envelope([choice(output, output.delta)]))}\n\n'
                    if final is not None and (body.stream_options or {}).get('include_usage'):
                        chunk = envelope([])
                        chunk['usage'] = usage(final)
                        yield f'data: {json.dumps(chunk)}\n\n'
                except (ValueError, TypeError, RuntimeError):
                    yield 'data: {"error":{"message":"Generation failed","type":"server_error"}}\n\n'
                yield 'data: [DONE]\n\n'
            return StreamingResponse(events(), media_type='text/event-stream',
                                     headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        try:
            final = None
            async with aclosing(engine.generate(ids, params)) as outputs:
                async for output in outputs:
                    if await request.is_disconnected():
                        return error('Client disconnected', 499)
                    final = output
            if final is None:
                return error('Generation produced no output', 500, 'server_error')
            result = envelope([choice(final, final.text)])
            result['usage'] = usage(final)
            return JSONResponse(result)
        except (ValueError, TypeError) as exc:
            return error(str(exc))
        except RuntimeError:
            return error('Generation worker is unavailable', 503, 'server_error')

    @app.get('/health')
    async def health():
        if not engine.healthy:
            return error('Generation worker is unavailable', 503, 'server_error')
        return {'status': 'ok'}

    @app.get('/v1/models')
    async def models():
        return {'object': 'list', 'data': [{'id': model_name, 'object': 'model', 'created': 0,
                                           'owned_by': 'Tiny-vLLM', 'max_model_len': eng.cfg.max_model_len}]}

    @app.get('/metrics')
    async def metrics():
        return {'running': len(eng.scheduler.running), 'waiting': len(eng.scheduler.waiting),
                'free_kv_blocks': eng.kv.num_free, 'healthy': engine.healthy}

    @app.post('/v1/completions')
    async def completions(body: CompletionBody, request: Request):
        return await respond(body.prompt, body, False, request)

    @app.post('/v1/chat/completions')
    async def chat(body: ChatBody, request: Request):
        messages = [{'role': m.role, 'content': m.content if isinstance(m.content, str)
                     else ''.join(part.text for part in m.content)} for m in body.messages]
        kwargs = body.chat_template_kwargs or {}
        if {'tokenize', 'add_generation_prompt', 'conversation'} & kwargs.keys():
            return error('Reserved chat-template arguments are not supported')
        try:
            prompt = eng.tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, **kwargs)
        except (ValueError, TypeError) as exc:
            return error(str(exc))
        return await respond(prompt, body, True, request)

    return app

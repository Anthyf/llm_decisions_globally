from abc import ABC, abstractmethod
from openai import OpenAI
from openai._types import NOT_GIVEN
from llamaapi import LlamaAPI
from google import genai
from google.genai import types
import anthropic
from anthropic import AsyncAnthropic
import json
import asyncio
from openai import AsyncOpenAI

class LLMWrapper(ABC):
    @abstractmethod
    def generate_response(self, prompt):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass

class AsyncOpenRouterApi(LLMWrapper):
    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
        provider_order=None,
    ):
        self.system_role = system_role
        self.model = model
        self.temperature = temperature
        self.provider_order = provider_order
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.messages = []

    async def generate_response(self, prompt, history=None, stream=False):
        if history is None:
            history = []

        messages = history + [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]

        provider = {"allow_fallbacks": False}
        if self.provider_order is not None:
            provider["order"] = self.provider_order

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=stream,
            max_tokens=1000,
            extra_body={"provider": provider},
        )

        if stream:
            return completion
        else:
            return completion.choices[0].message.content


class AsyncGptApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
        # provider_order=None,
        **kwargs, 
    ):
        self.system_role = system_role
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)
        self.messages = []
        self.temperature = temperature

    async def generate_response(self, prompt):

        messages = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": prompt},
        ]
        completion = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )

        return completion.choices[0].message.content



class AsyncClaudeApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
        # provider_order=None,
        **kwargs,
    ):

        self.system_role = system_role
        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)
        self.messages = []
        self.temperature = temperature

    async def generate_response(self, prompt):

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            system=self.system_role,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )

        return message.content[0].text


class AsyncMistralApi(LLMWrapper):
    def __init__(
        self,
        api_key,
        model,
        system_role="You are an AI assistant. You make good decisions on behalf of the human",
        temperature=0.6,
        max_tokens=1000,
        **kwargs,  # swallows extras like provider_order so it's a drop-in swap
    ):
        self.system_role = system_role
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url="https://api.mistral.ai/v1", api_key=api_key)
        self.messages = []

    async def generate_response(self, prompt, history=None, stream=False):
        if history is None:
            history = []

        messages = (
            [{"role": "system", "content": self.system_role}]
            + history
            + [{"role": "user", "content": prompt}]
        )

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=stream,
            max_tokens=self.max_tokens,
        )

        if stream:
            return completion
        return completion.choices[0].message.content

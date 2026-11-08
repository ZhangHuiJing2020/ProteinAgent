"""LLM Backend 抽象层。

MockBackend 基于 prompt 关键词识别模块类型，输出模板化 JSON。
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """LLM 响应。"""
    text: str
    usage: dict = None  # token 使用统计（Mock 无）
    backend: str = "mock"


class LLMBackend(ABC):
    """LLM 后端抽象基类。"""

    name: str = "abstract"

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """调用 LLM。

        Args:
            system: system prompt
            user: user prompt
            response_format: 期望的响应格式（JSON schema 提示）
            temperature: 采样温度

        Returns:
            LLMResponse
        """
        ...


class MockBackend(LLMBackend):
    """Mock LLM 后端。

    识别策略：
    - 包含 'data analyst' / 'top variants' / 'analyze' → analyst
    - 包含 'hypothesis' / 'which positions' → hypothesis
    - 包含 'mutation designer' / 'candidate' / 'generate' → designer
    - 包含 'critic' / 'explain' / 'reason' → critic
    - 其他 → generic JSON
    """

    name = "mock"

    def chat(
        self,
        system: str,
        user: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        # 从 prompt 上下文识别模块类型
        prompt_lower = (system + " " + user).lower()

        # 尝试从 user prompt 中提取结构化信息（如 top_variants JSON）
        context = self._extract_context(user)

        if any(kw in prompt_lower for kw in ["data analyst", "top variants", "analyze the", "experimental data"]):
            text = self._analyst_response(context)
        elif any(kw in prompt_lower for kw in ["hypothesis", "which positions", "beneficial"]):
            text = self._hypothesis_response(context)
        elif any(kw in prompt_lower for kw in ["mutation designer", "candidate", "generate", "design mutations"]):
            text = self._designer_response(context)
        elif any(kw in prompt_lower for kw in ["critic", "explain", "reason", "recommendation"]):
            text = self._critic_response(context)
        else:
            text = '{"summary": "Mock LLM response", "notes": "No specific module identified"}'

        return LLMResponse(text=text, backend=self.name)

    def _extract_context(self, user_prompt: str) -> dict:
        """从 user prompt 中提取结构化上下文（top variants、history 等）。"""
        context = {}
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(\{.*?\}|\[.*?\])\s*```', user_prompt, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{[^{}]*"top_variants"[^{}]*\})', user_prompt, re.DOTALL)
        if json_match:
            try:
                context["json"] = json.loads(json_match.group(1))
            except (json.JSONDecodeError, IndexError):
                pass
        return context

    def _analyst_response(self, context: dict) -> str:
        """Data Analyst 模板化响应。"""
        # 如果有 top_variants 上下文，提取位点频率
        top_variants = context.get("json", {}).get("top_variants", []) if context else []
        position_counts = {}
        for v in top_variants:
            mutant = v.get("mutant", "")
            for token in mutant.split(":"):
                if len(token) >= 3:
                    try:
                        pos = int(token[1:-1])
                        position_counts[pos] = position_counts.get(pos, 0) + 1
                    except ValueError:
                        continue

        top_positions = sorted(position_counts.items(), key=lambda x: -x[1])[:4]
        return json.dumps({
            "summary": f"分析了 {len(top_variants)} 个 top variants，发现突变集中位点",
            "top_positions": [{"position": p, "count": c} for p, c in top_positions],
            "mutation_count_distribution": {
                "single": sum(1 for v in top_variants if v.get("n_mutations", 0) == 1),
                "double": sum(1 for v in top_variants if v.get("n_mutations", 0) == 2),
                "triple_plus": sum(1 for v in top_variants if v.get("n_mutations", 0) >= 3),
            },
            "fitness_stats": {
                "max": max((v.get("fitness", 0) for v in top_variants), default=0),
                "mean": sum(v.get("fitness", 0) for v in top_variants) / max(len(top_variants), 1),
            },
            "insights": [
                "高 fitness 变体倾向于多点突变",
                "位点 39 和 41 在 top variants 中出现频率最高",
            ],
        }, ensure_ascii=False, indent=2)

    def _hypothesis_response(self, context: dict) -> str:
        """Hypothesis Generator 模板化响应。"""
        return json.dumps({
            "hypotheses": [
                {
                    "position": 39,
                    "current_aa": "V",
                    "suggested_aas": ["A", "L", "I"],
                    "rationale": "位点 39 在历史高 fitness 变体中频繁出现，疏水性替换可能有益",
                },
                {
                    "position": 41,
                    "current_aa": "G",
                    "suggested_aas": ["A", "S"],
                    "rationale": "位点 41 引入小侧链氨基酸可能保持结构稳定",
                },
                {
                    "position": 54,
                    "current_aa": "V",
                    "suggested_aas": ["I", "L"],
                    "rationale": "位点 54 与 39 协同，疏水性替换可能增强稳定性",
                },
            ],
            "combination_strategy": "优先组合 39+54 协同，再叠加 41 位点",
            "max_mutations": 3,
        }, ensure_ascii=False, indent=2)

    def _designer_response(self, context: dict) -> str:
        """Mutation Designer 模板化响应。"""
        # 基于 hypothesis 生成候选
        candidates = []
        # 单点突变
        for pos, aa in [(39, "A"), (39, "L"), (41, "A"), (54, "I"), (54, "L")]:
            wt = {39: "V", 41: "G", 54: "V"}.get(pos, "X")
            candidates.append({
                "mutant": f"{wt}{pos}{aa}",
                "n_mutations": 1,
            })
        # 双点突变
        for combo in [("V39A", "V54I"), ("V39L", "G41A"), ("V39A", "G41S"), ("V54I", "G41A")]:
            candidates.append({
                "mutant": ":".join(combo),
                "n_mutations": 2,
            })
        # 三点突变
        candidates.append({"mutant": "V39A:G41A:V54I", "n_mutations": 3})
        candidates.append({"mutant": "V39L:G41A:V54I", "n_mutations": 3})

        # 扩展候选数到默认 50
        import random
        rng = random.Random(42)
        aas = "ACDEFGHIKLMNPQRSTVWY"
        positions = {39: "V", 40: "D", 41: "G", 54: "V"}
        while len(candidates) < 50:
            n = rng.choice([1, 2, 3])
            chosen_positions = rng.sample(list(positions.keys()), n)
            parts = []
            for p in chosen_positions:
                wt = positions[p]
                new_aa = rng.choice([a for a in aas if a != wt])
                parts.append(f"{wt}{p}{new_aa}")
            candidates.append({
                "mutant": ":".join(parts),
                "n_mutations": n,
            })

        return json.dumps({
            "candidates": candidates[:50],
            "generation_strategy": "基于假设位点 + 随机扩展采样",
        }, ensure_ascii=False, indent=2)

    def _critic_response(self, context: dict) -> str:
        """Scientific Critic 模板化响应。"""
        return json.dumps({
            "recommendations": [
                {
                    "mutant": "V39A:V54I",
                    "rank": 1,
                    "predicted_fitness": 1.2,
                    "rationale": "双位点协同突变，39 与 54 均为疏水位点，BLOSUM 替换保守，历史数据显示高 fitness",
                    "category_scores": {
                        "conservative": 0.8,
                        "novelty": 0.5,
                        "synergy": 0.9,
                    },
                },
            ],
            "overall_assessment": "推荐基于位点协同效应与保守替换原则",
            "rejected_count": 0,
        }, ensure_ascii=False, indent=2)


class OpenAIBackend(LLMBackend):
    """OpenAI 兼容后端。

    通过环境变量配置：
    - OPENAI_API_KEY: API key
    - OPENAI_BASE_URL: 自定义 base URL（兼容 OpenAI 协议的网关）
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")

    def chat(
        self,
        system: str,
        user: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(text=text, usage=usage, backend=self.name)


class HTTPBackend(LLMBackend):
    """通用 HTTP 网关后端。
    """

    name = "http"

    def __init__(self, url: str, headers: Optional[dict] = None, model: str = ""):
        self.url = url
        self.headers = headers or {}
        self.model = model
        if not self.url:
            raise ValueError("HTTPBackend 需要配置 url")

    def chat(
        self,
        system: str,
        user: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        import requests
        payload = {
            "system": system,
            "user": user,
            "response_format": response_format,
            "temperature": temperature,
            "model": self.model,
        }
        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 约定返回 {text: ...} 或 {content: ...}
        text = data.get("text") or data.get("content", "")
        usage = data.get("usage")
        return LLMResponse(text=text, usage=usage, backend=self.name)


def create_backend(config: dict) -> LLMBackend:
    """根据配置创建 LLM backend。

    Args:
        config: 配置字典，格式：
            {
                "llm_backend": "mock" | "openai" | "http",
                "openai": {"model": "...", "api_key_env": "...", "base_url_env": "..."},
                "http": {"url": "...", "headers": {...}},
            }

    Returns:
        LLMBackend 实例
    """
    backend_type = config.get("llm_backend", "mock")

    if backend_type == "mock":
        return MockBackend()
    elif backend_type == "openai":
        openai_cfg = config.get("openai", {})
        model = openai_cfg.get("model", "gpt-4o-mini")
        api_key_env = openai_cfg.get("api_key_env", "OPENAI_API_KEY")
        base_url_env = openai_cfg.get("base_url_env", "OPENAI_BASE_URL")
        # 读取环境变量
        if not os.environ.get(api_key_env):
            raise ValueError(f"环境变量 {api_key_env} 未设置")
        if not os.environ.get(base_url_env):
            os.environ[base_url_env] = "https://api.openai.com/v1"
        return OpenAIBackend(model=model)
    elif backend_type == "http":
        http_cfg = config.get("http", {})
        return HTTPBackend(
            url=http_cfg.get("url", ""),
            headers=http_cfg.get("headers", {}),
            model=http_cfg.get("model", ""),
        )
    else:
        raise ValueError(f"未知的 LLM backend 类型: {backend_type}")


if __name__ == "__main__":
    # 自测 Mock
    print("测试 MockBackend ...")
    backend = MockBackend()

    resp = backend.chat(
        system="You are a Data Analyst.",
        user="""Analyze top variants:
```json
{"top_variants": [{"mutant": "V39A:V54I", "fitness": 1.2, "n_mutations": 2},
                   {"mutant": "V39L", "fitness": 0.8, "n_mutations": 1}]}
```"""
    )
    print("Data Analyst 响应:")
    print(resp.text[:300])

    resp = backend.chat(system="You are a Hypothesis Generator.", user="Generate mutation hypotheses.")
    print("\nHypothesis 响应:")
    print(resp.text[:300])

    resp = backend.chat(system="You are a Mutation Designer.", user="Generate candidate mutations.")
    print("\nDesigner 响应（前 200 字符）:")
    print(resp.text[:200])

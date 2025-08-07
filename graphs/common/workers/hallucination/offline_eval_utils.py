from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)

from typing import Any, List, Dict, Optional
import json
import ast
from pydantic import BaseModel


class OfflineHallucinationAnalysis(BaseModel):
    call_id: str
    turn_id: int
    agent_name: Optional[str] = None
    human_input: Optional[str] = None
    agent_text_output: Optional[str] = None
    judge_label: Optional[str] = None
    true_label: Optional[str] = None
    comments: Optional[str] = None
    explanation: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []


def raw_to_json(messages_data) -> List[Dict[str, Any]]:
    if isinstance(messages_data, str):
        if messages_data.strip().startswith(("[", "{")):
            try:
                return json.loads(messages_data)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return ast.literal_eval(messages_data)
            except (ValueError, SyntaxError):
                pass
    elif isinstance(messages_data, list):
        collect = []
        for item in messages_data:
            if isinstance(item, str):
                if item.strip().startswith(("[", "{")):
                    try:
                        parsed = json.loads(item)
                        collect.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    try:
                        parsed = ast.literal_eval(item)
                        collect.append(parsed)
                    except (ValueError, SyntaxError):
                        pass
                else:
                    # If it's a string but not JSON, skip it
                    raise ValueError("json parsing failed 1")
            elif isinstance(item, dict):
                collect.append(item)
            else:
                raise ValueError("json parsing failed 2")
        return collect

    return []


def serialize_message(msg: BaseMessage) -> dict:
    msg_dict = {
        "type": msg.type,
        "content": msg.content,
        "id": getattr(msg, "id", None),
    }

    # put tool_calls back for AIMessages
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        msg_dict["tool_calls"] = msg.tool_calls

    # put tool_call_id and name back for ToolMessages
    if isinstance(msg, ToolMessage):
        msg_dict["tool_call_id"] = msg.tool_call_id
        msg_dict["name"] = msg.name

    # re-inject any extra fields you stored
    for k, v in getattr(msg, "additional_kwargs", {}).items():
        msg_dict[k] = v

    return msg_dict


def compute_metrics(analyses: List[OfflineHallucinationAnalysis]) -> Dict[str, float]:
    y_true = []
    y_pred = []

    for analysis in analyses:
        true_label = analysis.true_label
        judge_label = analysis.judge_label

        if not true_label or not isinstance(true_label, str):
            continue

        true_label_lower = true_label.lower()
        if true_label_lower not in ["positive", "negative"]:
            continue

        if not judge_label or not isinstance(judge_label, str):
            continue

        y_true.append(1 if true_label_lower == "positive" else 0)
        judge_label_lower = judge_label.lower()
        y_pred.append(1 if judge_label_lower == "positive" else 0)

    if not y_true:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "valid_count": 0,
        }

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "valid_count": len(y_true),
    }

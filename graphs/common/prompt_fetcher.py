import os
from dotenv import load_dotenv
import logging
from typing import Optional, Dict
import re

from langchain_core.prompts import ChatPromptTemplate
from jinja2 import Template

from app_config import AppConfig
from post_call_processors.transcript import replace_client_name_in_text
from redis_config_manager import RedisConfigManager
from utils.feature_flag.client import FeatureFlag

PROMPTS_TABLE = "chat_prompts"

config_manager = RedisConfigManager()
logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.local")
PROMPT_IDS = os.getenv("PROMPT_IDS")

# Parse the PROMPT_IDS env variable into a dictionary
PROMPT_ID_MAPPING: Dict[str, str] = {}
if PROMPT_IDS:
    for pair in PROMPT_IDS.split(","):
        if "=" in pair:
            name, id_value = pair.split("=")
            PROMPT_ID_MAPPING[name.strip()] = id_value.strip()


def get_prompt(
    name: str,
    language: str,
    client: str,
    call_type: Optional[str] = None,
    config: Optional[dict] = None,
    return_raw_prompt: bool = False,
    prompt_id: Optional[str] = None,
):
    # Check if the prompt name is in our mapping and use that ID if available
    if name in PROMPT_ID_MAPPING and not prompt_id:
        prompt_id = PROMPT_ID_MAPPING[name]
        logger.info(f"Using prompt ID {prompt_id} from environment for {name}")

    if call_type == "inbound":
        call_type = "collections"

    call_id = AppConfig().call_metadata.get("call_id")
    logger.info(f"Call ID: {call_id}")

    prompt_variation = AppConfig().feature_flag_client.is_feature_enabled(
        FeatureFlag.PROMPT_TESTING_ENABLED, client, call_id
    )

    logger.info(f"Prompt variation: {prompt_variation}")

    # Construct redis key based on available parameters
    redis_key = f"prompt:{name}:{language}:{client}:{call_type}:"
    if prompt_id:
        redis_key += f"{prompt_id}:"
    redis_key += f"{prompt_variation}"

    # Try to get cached prompt first
    prompt_template = config_manager.get_value(redis_key)

    if not prompt_template:
        query = AppConfig().supabase.from_(PROMPTS_TABLE).select("*")

        # Priority 1: Fetch by prompt_id if available
        if prompt_id:
            response = query.eq("id", prompt_id).execute()
            if response.data:
                prompt_template = response.data[0]["prompt"]
                config_manager.set_value(redis_key, prompt_template, ttl=600)

        # Priority 2: If still no template and prompt_variation is testing-prompt
        elif prompt_variation == "testing-prompt" and not prompt_template:
            query = (
                query.eq("name", name)
                .eq("language", language)
                .eq("tenant", client)
                .eq("is_test_variant", True)
                .eq("is_self_serve", False)
            )
            if call_type:
                query = query.eq("call_type", call_type)

            response = query.execute()
            if response.data:
                logger.info("Found testing prompt variant")
                prompt_template = response.data[0]["prompt"]
                config_manager.set_value(redis_key, prompt_template, ttl=600)

        # Priority 3: Fetch regular published prompt if still no template
        if not prompt_template:
            query = (
                AppConfig()
                .supabase.from_(PROMPTS_TABLE)
                .select("*")
                .eq("name", name)
                .eq("language", language)
                .eq("tenant", client)
                .eq("is_published", True)
                .eq("is_self_serve", False)
            )

            if call_type:
                query = query.eq("call_type", call_type)

            response = query.order("version", desc=True).execute()

            if not response.data:
                raise ValueError(
                    f"No prompt found for agent:{name} in {language} for client {client} and call type {call_type}"
                )

            if len(response.data) > 1:
                logger.warning(
                    f"Multiple published prompts found for agent {name} in {language} for client {client} and call type {call_type}"
                )

            # Print all attributes except the prompt text itself
            prompt_data = response.data[0]
            filtered_data = {
                k: v for k, v in prompt_data.items() if k != "prompt"
            }
            logger.info(f"Prompt metadata: {filtered_data}")

            prompt_template = response.data[0]["prompt"]
            logger.info(f"Prompt template: {prompt_template}")
            config_manager.set_value(redis_key, prompt_template, ttl=600)

    if config:
        # Regex: replace {variable} with {{ variable }} only if not already in {{ ... }} or {% ... %}
        def curly_to_jinja(text):
            pattern = r'(?<![{%])\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
            return re.sub(pattern, r'{{ \1 }}', text)
        jinja_template = curly_to_jinja(prompt_template)
        template = Template(jinja_template)
        prompt = template.render(**config)
    else:
        prompt = prompt_template

    prompt = replace_client_name_in_text(prompt)

    if return_raw_prompt:
        return prompt

    return ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
        ]
    )

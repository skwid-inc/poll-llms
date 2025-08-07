import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Dict

from app_config import AppConfig
from utils.logger import logger


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON serializable format.
    Handles AIMessage objects and other non-serializable types.
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        # If object has a to_dict method (like AIMessage), use it
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        # For objects with __dict__, convert to dict and recursively process values
        result = {}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):  # Skip private attributes
                result[key] = _make_json_serializable(value)
        return result
    elif isinstance(obj, dict):
        # Handle dictionaries by recursively converting their values
        return {
            key: _make_json_serializable(value) for key, value in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        # Handle lists and tuples by recursively converting their items
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Basic types are already serializable
        return obj
    else:
        # For other types, convert to string
        return str(obj)


class AsyncSupabaseLogger:
    def __init__(self, max_workers: int = 3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._queue: asyncio.Queue[(str, Dict[str, Any])] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the background processing task"""
        logger.info(f"async logger is_started: {self.is_started()}")
        if not self.is_started():
            logger.info("Starting async logger queue consumer")
            self._task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stop the background processing task and wait for queue to empty"""
        if self._task is not None:
            await self._queue.join()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def is_started(self):
        return self._task is not None

    def _save_to_supabase(
        self,
        args: Dict[str, Any],
        table_name: str = "langgraph_prompt_completion",
    ) -> None:
        """Synchronous function to save to Supabase"""
        try:
            # Convert args to JSON serializable format
            serializable_args = _make_json_serializable(args)
            AppConfig().supabase.table(table_name).insert(
                serializable_args
            ).execute()
        except Exception as e:
            logger.error(f"Error saving to Supabase table {table_name}: {e}")

    async def _process_queue(self):
        """Background task to process the queue"""
        while True:
            table_name, args = await self._queue.get()
            # logger.info(f"Processing table: {table_name}, args: {args}")
            try:
                # Run the blocking Supabase operation in the thread pool
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    partial(self._save_to_supabase, args, table_name),
                )
            finally:
                self._queue.task_done()

    async def save_prompt_completion(self, args: Dict[str, Any]) -> None:
        """Async function to queue a save operation"""
        await self._queue.put(("langgraph_prompt_completion", args))

    async def queue_save_to_supabase(
        self,
        args: Dict[str, Any],
        table_name: str = "langgraph_prompt_completion",
    ) -> None:
        """Async function to queue a save operation"""
        # logger.info(
        #     f"Queuing save to Supabase table: {table_name}, args: {args}"
        # )
        await self._queue.put((table_name, args))

    async def write_to_supabase(
        self,
        args: Dict[str, Any],
        table_name: str = "langgraph_prompt_completion",
    ) -> None:
        """Async function to write to Supabase"""
        await self.start()
        await self.queue_save_to_supabase(args, table_name)


class AsyncSupabaseLoggerForGuardrails(AsyncSupabaseLogger):
    async def save_guardrails_output(self, args: Dict[str, Any]) -> None:
        """Async function to queue a save operation"""
        print(f"[GuardRails] Table name: guardrails_output")
        await self._queue.put(args)

    def _save_to_supabase(self, args: Dict[str, Any]) -> None:
        """Synchronous function to save to Supabase"""
        try:
            serializable_args = _make_json_serializable(args)
            AppConfig().supabase.table("guardrails_output").insert(
                serializable_args
            ).execute()
        except Exception as e:
            print(f"Error saving guardrails output to Supabase: {e}")

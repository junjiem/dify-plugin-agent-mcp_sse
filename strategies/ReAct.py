import time
from collections.abc import Generator, Mapping
from typing import Any, Optional, cast

import orjson
import pydantic
from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
    PromptMessageTool, PromptMessageContentType,
)
from dify_plugin.entities.tool import (
    LogMetadata,
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentScratchpadUnit,
    AgentStrategy,
    ToolEntity,
)
from pydantic import BaseModel

from output_parser.cot_output_parser import CotAgentOutputParser
from prompt.template import REACT_PROMPT_TEMPLATES
from utils.mcp_client import McpClients

ignore_observation_providers = ["wenxin"]


class ReActParams(BaseModel):
    query: str
    instruction: str
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    mcp_servers_config: str | None
    mcp_resources_as_tools: bool = False
    mcp_prompts_as_tools: bool = False
    maximum_iterations: int = 3


class AgentPromptEntity(BaseModel):
    """
    Agent Prompt Entity.
    """

    first_prompt: str
    next_iteration: str


class ReActAgentStrategy(AgentStrategy):
    def __init__(self, runtime, session):
        super().__init__(runtime, session)
        self.query = ""
        self.instruction = ""
        self.history_prompt_messages = []
        self.prompt_messages_tools = []

    @property
    def _user_prompt_message(self) -> UserPromptMessage:
        return UserPromptMessage(content=self.query)

    @property
    def _system_prompt_message(self) -> SystemPromptMessage:
        prompt_entity = AgentPromptEntity(
            first_prompt=REACT_PROMPT_TEMPLATES["english"]["chat"]["prompt"],
            next_iteration=REACT_PROMPT_TEMPLATES["english"]["chat"][
                "agent_scratchpad"
            ],
        )
        if not prompt_entity:
            raise ValueError("Agent prompt configuration is not set")
        first_prompt = prompt_entity.first_prompt

        system_prompt = (
            first_prompt.replace("{{instruction}}", self.instruction)
            .replace(
                "{{tools}}",
                orjson.dumps(
                    [
                        tool.model_dump(mode="json")
                        for tool in self._prompt_messages_tools
                    ],
                ).decode('utf-8'),
            )
            .replace(
                "{{tool_names}}",
                ", ".join([tool.name for tool in self._prompt_messages_tools]),
            )
        )

        return SystemPromptMessage(content=system_prompt)

    def _iter_cleanup_history_prompt_messages(self, model: AgentModelConfig):
        """
        remove history_prompt_message if model not support
        :param model
        :return:
        """
        for msg in model.history_prompt_messages:
            if isinstance(msg.content, list):
                filtered_content = [
                    item
                    for item in msg.content
                    if (
                            item.type == PromptMessageContentType.TEXT
                            or (item.type in {
                                PromptMessageContentType.IMAGE, PromptMessageContentType.VIDEO,
                                PromptMessageContentType.DOCUMENT,
                            } and ModelFeature.VISION in model.entity.features)
                            or (item.type == PromptMessageContentType.AUDIO and ModelFeature.AUDIO in model.entity.features)
                            or (item.type == PromptMessageContentType.VIDEO and ModelFeature.VIDEO in model.entity.features)
                            or (item.type == PromptMessageContentType.DOCUMENT and ModelFeature.DOCUMENT in model.entity.features)
                    )
                ]
                new_msg = msg.__class__(
                    role=msg.role,
                    content=filtered_content,
                    name=msg.name,
                )
                yield new_msg
            else:
                yield msg


    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        """
        Run ReAct agent application
        """

        try:
            react_params = ReActParams(**parameters)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid parameters: {e!s}") from e

        # Init parameters
        self.query = react_params.query
        self.instruction = react_params.instruction or self.instruction
        agent_scratchpad = []
        iteration_step = 1
        max_iteration_steps = react_params.maximum_iterations
        run_agent_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""
        prompt_messages = []

        # Init model
        model = react_params.model
        stop = (
            react_params.model.completion_params.get("stop", [])
            if react_params.model.completion_params
            else []
        )
        if (
                "Observation" not in stop
                and model.provider not in ignore_observation_providers
        ):
            stop.append("Observation")

        # Init prompts
        self.history_prompt_messages = list(self._iter_cleanup_history_prompt_messages(model))


        # convert tools into ModelRuntime Tool format
        tools = react_params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}

        # Fetch MCP tools
        mcp_clients = None
        mcp_tools = []
        mcp_tool_instances = {}
        mcp_servers_config = react_params.mcp_servers_config
        mcp_resources_as_tools = react_params.mcp_resources_as_tools
        mcp_prompts_as_tools = react_params.mcp_prompts_as_tools
        if mcp_servers_config:
            try:
                # Injected variable mcp_servers_config begin and end has double quotes.
                servers_config = orjson.loads(mcp_servers_config.strip('"'))
            except orjson.JSONDecodeError as e:
                raise ValueError(f"mcp_servers_config must be a valid JSON string: {e}")
            mcp_clients = McpClients(servers_config, mcp_resources_as_tools, mcp_prompts_as_tools)
            mcp_tools = mcp_clients.fetch_tools()
            mcp_tool_instances = {tool.get("name"): tool for tool in mcp_tools} if mcp_tools else {}

        react_params.model.completion_params = (
                react_params.model.completion_params or {}
        )
        # convert tools into ModelRuntime Tool format
        prompt_messages_tools = self._init_prompt_tools(tools)
        prompt_messages_tools.extend(self._init_prompt_mcp_tools(mcp_tools))
        self._prompt_messages_tools = prompt_messages_tools

        while run_agent_state and iteration_step <= max_iteration_steps:
            # continue to run until there is not any tool call
            run_agent_state = False
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log
            if iteration_step == max_iteration_steps:
                # the last iteration, remove all tools
                self._prompt_messages_tools = []

            message_file_ids: list[str] = []

            # recalc llm max tokens
            prompt_messages = self._organize_prompt_messages(
                agent_scratchpad, self.query
            )
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity, prompt_messages, model.completion_params
                )

            # prevent Claude LLM errors caused by extra whitespace.
            for i, msg in enumerate(prompt_messages):
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    msg.content = msg.content.strip()

            # invoke model
            chunks = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop,
            )

            usage_dict = {}
            react_chunks = CotAgentOutputParser.handle_react_stream_output(
                chunks, usage_dict
            )
            scratchpad = AgentScratchpadUnit(
                agent_response="",
                thought="",
                action_str="",
                observation="",
                action=None,
            )

            model_started_at = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            for chunk in react_chunks:
                if isinstance(chunk, AgentScratchpadUnit.Action):
                    action = chunk
                    # detect action
                    assert scratchpad.agent_response is not None
                    scratchpad.agent_response += orjson.dumps(chunk.model_dump()).decode('utf-8')

                    scratchpad.action_str = orjson.dumps(chunk.model_dump()).decode('utf-8')
                    scratchpad.action = action
                else:
                    scratchpad.agent_response = scratchpad.agent_response or ""
                    scratchpad.thought = scratchpad.thought or ""
                    scratchpad.agent_response += chunk
                    scratchpad.thought += chunk
            scratchpad.thought = (
                scratchpad.thought.strip()
                if scratchpad.thought
                else "I am thinking about how to help you"
            )
            agent_scratchpad.append(scratchpad)

            # get llm usage
            if "usage" in usage_dict:
                if usage_dict["usage"] is not None:
                    self.increase_usage(llm_usage, usage_dict["usage"])
            else:
                usage_dict["usage"] = LLMUsage.empty_usage()

            action = (
                scratchpad.action.to_dict()
                if scratchpad.action
                else {"action": scratchpad.agent_response}
            )

            yield self.finish_log_message(
                log=model_log,
                data={"thought": scratchpad.thought, **action},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            if not scratchpad.action:
                final_answer = scratchpad.thought
            else:
                if scratchpad.action.action_name.lower() == "final answer":
                    # action is final answer, return final answer directly
                    try:
                        if isinstance(scratchpad.action.action_input, dict):
                            final_answer = orjson.dumps(scratchpad.action.action_input).decode('utf-8')
                        elif isinstance(scratchpad.action.action_input, str):
                            final_answer = scratchpad.action.action_input
                        else:
                            final_answer = f"{scratchpad.action.action_input}"
                    except orjson.JSONDecodeError:
                        final_answer = f"{scratchpad.action.action_input}"
                else:
                    run_agent_state = True
                    # action is tool call, invoke tool
                    tool_call_started_at = time.perf_counter()
                    tool_name = scratchpad.action.action_name
                    tool_call_log = self.create_log_message(
                        label=f"CALL {tool_name}",
                        data={},
                        metadata={
                            LogMetadata.STARTED_AT: time.perf_counter(),
                            LogMetadata.PROVIDER: tool_instances[
                                tool_name
                            ].identity.provider
                            if tool_instances.get(tool_name)
                            else "",
                        },
                        parent=round_log,
                        status=ToolInvokeMessage.LogMessage.LogStatus.START,
                    )
                    yield tool_call_log
                    tool_invoke_response, tool_invoke_parameters = (
                        self._handle_invoke_action(
                            action=scratchpad.action,
                            tool_instances=tool_instances,
                            mcp_clients=mcp_clients,
                            mcp_tool_instances=mcp_tool_instances,
                            message_file_ids=message_file_ids,
                        )
                    )
                    scratchpad.observation = tool_invoke_response
                    scratchpad.agent_response = tool_invoke_response
                    yield self.finish_log_message(
                        log=tool_call_log,
                        data={
                            "tool_name": tool_name,
                            "tool_call_args": tool_invoke_parameters,
                            "output": tool_invoke_response,
                        },
                        metadata={
                            LogMetadata.STARTED_AT: tool_call_started_at,
                            LogMetadata.PROVIDER: tool_instances[
                                tool_name
                            ].identity.provider
                            if tool_instances.get(tool_name)
                            else "",
                            LogMetadata.FINISHED_AT: time.perf_counter(),
                            LogMetadata.ELAPSED_TIME: time.perf_counter()
                                                      - tool_call_started_at,
                        },
                    )

                # update prompt tool message
                for prompt_tool in self._prompt_messages_tools:
                    if prompt_tool.name in tool_instances:
                        self.update_prompt_message_tool(
                            tool_instances[prompt_tool.name], prompt_tool
                        )
            yield self.finish_log_message(
                log=round_log,
                data={
                    "action_name": scratchpad.action.action_name
                    if scratchpad.action
                    else "",
                    "action_input": scratchpad.action.action_input
                    if scratchpad.action
                    else "",
                    "thought": scratchpad.thought,
                    "observation": scratchpad.observation,
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            iteration_step += 1

        # All MCP Client close
        if mcp_clients:
            mcp_clients.close()

        yield self.create_text_message(final_answer)
        yield self.create_json_message(
            {
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: llm_usage["usage"].total_price
                    if llm_usage["usage"] is not None
                    else 0,
                    LogMetadata.CURRENCY: llm_usage["usage"].currency
                    if llm_usage["usage"] is not None
                    else "",
                    LogMetadata.TOTAL_TOKENS: llm_usage["usage"].total_tokens
                    if llm_usage["usage"] is not None
                    else 0,
                }
            }
        )

    def _organize_user_query(
            self, query, prompt_messages: list[PromptMessage]
    ) -> list[PromptMessage]:
        """
        Organize user query
        """
        prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _organize_prompt_messages(
            self, agent_scratchpad: list, query: str
    ) -> list[PromptMessage]:
        """
        Organize
        """
        # organize system prompt
        system_message = self._system_prompt_message

        # organize current assistant messages
        agent_scratchpad = agent_scratchpad
        if not agent_scratchpad:
            assistant_messages = []
        else:
            assistant_message = AssistantPromptMessage(content="")
            for unit in agent_scratchpad:
                if unit.is_final():
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Final Answer: {unit.agent_response}"
                else:
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Thought: {unit.thought}\n\n"
                    if unit.action_str:
                        assistant_message.content += f"Action: {unit.action_str}\n\n"
                    if unit.observation:
                        assistant_message.content += (
                            f"Observation: {unit.observation}\n\n"
                        )

            assistant_messages = [assistant_message]

        # query messages
        query_messages = self._organize_user_query(query, [])

        # organize historic prompt messages
        historic_messages = self.history_prompt_messages

        if assistant_messages:
            messages = [
                system_message,
                *historic_messages,
                *query_messages,
                *assistant_messages,
            ]
        else:
            messages = [system_message, *historic_messages, *query_messages]

        # join all messages
        return messages

    def _handle_invoke_action(
            self,
            action: AgentScratchpadUnit.Action,
            mcp_clients: McpClients | None,
            tool_instances: Mapping[str, ToolEntity],
            mcp_tool_instances: Mapping[str, dict],
            message_file_ids: list[str],
    ) -> tuple[str, dict[str, Any] | str]:
        """
        handle invoke action
        :param action: action
        :param mcp_clients: MCP Clients
        :param tool_instances: tool instances
        :param mcp_tool_instances: MCP tool instances
        :param message_file_ids: message file ids
        :return: observation, meta
        """
        # action is tool call, invoke tool
        tool_call_name = action.action_name
        tool_call_args = action.action_input
        tool_instance = tool_instances.get(tool_call_name)
        mcp_tool_instance = mcp_tool_instances.get(tool_call_name)

        if not tool_instance and not mcp_tool_instance:
            answer = f"there is not a tool named {tool_call_name}"
            return answer, tool_call_args

        if isinstance(tool_call_args, str):
            try:
                tool_call_args = orjson.loads(tool_call_args)
            except orjson.JSONDecodeError as e:
                params = [
                    param.name
                    for param in tool_instance.parameters
                    if param.form == ToolParameter.ToolParameterForm.LLM
                ]
                if len(params) > 1:
                    raise ValueError("tool call args is not a valid json string") from e
                tool_call_args = {params[0]: tool_call_args} if len(params) == 1 else {}

        tool_invoke_parameters = {}
        try:
            if mcp_tool_instance:
                # invoke MCP tool
                tool_invoke_parameters = tool_call_args
                content = mcp_clients.execute_tool(
                    tool_name=tool_call_name,
                    tool_args=tool_invoke_parameters,
                )
                if len(content) == 1:
                    item = content[0]
                    if item["type"] == "text":
                        result = item["text"]
                    elif item["type"] in ("image", "video"):
                        result = orjson.dumps(item).decode('utf-8')
                    elif item["type"] == "resource":
                        result = orjson.dumps(item['resource']).decode('utf-8')
                    else:
                        result = orjson.dumps(item).decode('utf-8')
                else:
                    result = orjson.dumps(content).decode('utf-8')
            else:
                # invoke tool
                tool_invoke_parameters = {**tool_instance.runtime_parameters, **tool_call_args}
                tool_invoke_responses = self.session.tool.invoke(
                    provider_type=ToolProviderType(tool_instance.provider_type),
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters=tool_invoke_parameters,
                )
                result = ""
                for response in tool_invoke_responses:
                    if response.type == ToolInvokeMessage.MessageType.TEXT:
                        result += cast(ToolInvokeMessage.TextMessage, response.message).text
                    elif response.type == ToolInvokeMessage.MessageType.LINK:
                        result += (
                                f"result link: {cast(ToolInvokeMessage.TextMessage, response.message).text}."
                                + " please tell user to check it."
                        )
                    elif response.type in {
                        ToolInvokeMessage.MessageType.IMAGE_LINK,
                        ToolInvokeMessage.MessageType.IMAGE,
                    }:
                        result += (
                                "image has been created and sent to user already, "
                                + "you do not need to create it, just tell the user to check it now."
                        )
                    elif response.type == ToolInvokeMessage.MessageType.JSON:
                        text = orjson.dumps(
                            cast(
                                ToolInvokeMessage.JsonMessage, response.message
                            ).json_object,
                        ).decode('utf-8')
                        result += f"tool response: {text}."
                    else:
                        result += f"tool response: {response.message!r}."
        except Exception as e:
            result = f"tool invoke error: {e!s}"

        return result, tool_invoke_parameters

    def _convert_dict_to_action(self, action: dict) -> AgentScratchpadUnit.Action:
        """
        convert dict to action
        """
        return AgentScratchpadUnit.Action(
            action_name=action["action"], action_input=action["action_input"]
        )

    def _format_assistant_message(
            self, agent_scratchpad: list[AgentScratchpadUnit]
    ) -> str:
        """
        format assistant message
        """
        message = ""
        for scratchpad in agent_scratchpad:
            if scratchpad.is_final():
                message += f"Final Answer: {scratchpad.agent_response}"
            else:
                message += f"Thought: {scratchpad.thought}\n\n"
                if scratchpad.action_str:
                    message += f"Action: {scratchpad.action_str}\n\n"
                if scratchpad.observation:
                    message += f"Observation: {scratchpad.observation}\n\n"

        return message

    @staticmethod
    def _init_prompt_mcp_tools(mcp_tools: list[dict]) -> list[PromptMessageTool]:
        """
        Initialize prompt message MCP tools
        """
        prompt_messages_tools = []

        for tool in mcp_tools:
            prompt_message = PromptMessageTool(
                name=tool.get("name"),
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema"),
            )
            prompt_messages_tools.append(prompt_message)

        return prompt_messages_tools

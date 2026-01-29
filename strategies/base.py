from typing import Generator

from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.message import PromptMessageContentType, PromptMessage
from dify_plugin.interfaces.agent import AgentModelConfig


class FilterHistoryMessageByModelFeaturesMixin:

    @staticmethod
    def _iter_cleanup_history_prompt_messages(model: AgentModelConfig) -> Generator[PromptMessage, None, None]:
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

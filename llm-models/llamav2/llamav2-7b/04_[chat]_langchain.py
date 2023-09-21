# Databricks notebook source
# MAGIC %md
# MAGIC # Load Llama-2-7b-chat-hf from LangChain on Databricks
# MAGIC
# MAGIC This example notebook shows how to wrap Databricks endpoints as LLMs in LangChain. It supports two endpoint types:
# MAGIC
# MAGIC - Serving endpoint, recommended for production and development. See `02_[chat]_mlflow_logging_inference` for how to create one.
# MAGIC - Cluster driver proxy app, recommended for iteractive development. See `03_[chat]_serve_driver_proxy` for how to create one.
# MAGIC
# MAGIC Environment tested:
# MAGIC - MLR: 13.2 ML
# MAGIC - Instance:
# MAGIC   - Wrapping a serving endpoint: `i3.xlarge` on AWS, `Standard_DS3_v2` on Azure
# MAGIC   - Wrapping a cluster driver proxy app: `g5.4xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure (same instance as the driver proxy app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrapping Databricks endpoints as LLMs in LangChain
# MAGIC - If the model is a serving endpoint, it requires a model serving endpoint (see `02_[chat]_mlflow_logging_inference` for how to create one) to be in the "Ready" state.
# MAGIC - If the model is a cluster driver proxy app, it requires the driver proxy app of the `03_[chat]_serve_driver_proxy` example notebook running.
# MAGIC   - If running a Databricks notebook attached to the same cluster that runs the app, you only need to specify the driver port to create a `Databricks` instance.
# MAGIC   - If running on different cluster, you can manually specify the cluster ID to use, as well as Databricks workspace hostname and personal access token.

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.llms import Databricks
def transform_input(**request):
    request["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request["prompt"]},
        ]
    request["stop"] = []
    return request
  
def transform_output(response):
  return response["candidates"][0]["message"]["content"]


# COMMAND ----------

# If using serving endpoint, the model serving endpoint is created in `02_[chat]_mlflow_logging_inference`
# llm = Databricks(endpoint_name='llama2-7b-chat',
#                  transform_input_fn=transform_input,
#                  transform_output_fn=transform_output,)

# If the model is a cluster driver proxy app on the same cluster, you only need to specify the driver port.
llm = Databricks(cluster_driver_port="7777",
                 transform_input_fn=transform_input,
                 transform_output_fn=transform_output,)

# If the model is a cluster driver proxy app on the different cluster, you need to provide the cluster id
# llm = Databricks(cluster_id="0000-000000-xxxxxxxx"
#                  cluster_driver_port="7777",
#                  transform_input_fn=transform_input,
#                  transform_output_fn=transform_output,)

print(llm("How to master Python in 3 days?"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap the model to a chat model
# MAGIC
# MAGIC We can define a langchain `ChatModel` with Databricks LLM interface so that it could be used in `LLMChain`.

# COMMAND ----------

from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.llms.databricks import (
    get_default_host,
    get_default_api_token,
    get_repl_context,
    _DatabricksClientBase,
    _DatabricksServingEndpointClient,
    _DatabricksClusterDriverProxyClient,
)
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatMLflowAIGateway
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    PrivateAttr,
    root_validator,
    validator,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)


class ChatParams(BaseModel, extra=Extra.allow):  # type: ignore[call-arg]
    """Parameters for the `MLflow AI Gateway` LLM."""

    temperature: float = 0.0
    candidate_count: int = 1
    """The number of candidates to return."""
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None


class ChatDatabricks(BaseChatModel):
    """
    chat model using Databricks LLM
    """
    host: str = Field(default_factory=get_default_host)
    api_token: str = Field(default_factory=get_default_api_token)
    endpoint_name: Optional[str] = None
    cluster_id: Optional[str] = None
    cluster_driver_port: Optional[str] = None

    _client: _DatabricksClientBase = PrivateAttr()
    params: Optional[ChatParams] = None

    @validator("cluster_id", always=True)
    def set_cluster_id(cls, v: Any, values: Dict[str, Any]) -> Optional[str]:
        if v and values["endpoint_name"]:
            raise ValueError("Cannot set both endpoint_name and cluster_id.")
        elif values["endpoint_name"]:
            return None
        elif v:
            return v
        else:
            try:
                if v := get_repl_context().clusterId:
                    return v
                raise ValueError("Context doesn't contain clusterId.")
            except Exception as e:
                raise ValueError(
                    "Neither endpoint_name nor cluster_id was set. "
                    "And the cluster_id cannot be automatically determined. Received"
                    f" error: {e}"
                )

    @validator("cluster_driver_port", always=True)
    def set_cluster_driver_port(cls, v: Any, values: Dict[str, Any]) -> Optional[str]:
        if v and values["endpoint_name"]:
            raise ValueError("Cannot set both endpoint_name and cluster_driver_port.")
        elif values["endpoint_name"]:
            return None
        elif v is None:
            raise ValueError(
                "Must set cluster_driver_port to connect to a cluster driver."
            )
        elif int(v) <= 0:
            raise ValueError(f"Invalid cluster_driver_port: {v}")
        else:
            return v

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.endpoint_name:
            self._client = _DatabricksServingEndpointClient(
                host=self.host,
                api_token=self.api_token,
                endpoint_name=self.endpoint_name,
            )
        elif self.cluster_id and self.cluster_driver_port:
            self._client = _DatabricksClusterDriverProxyClient(
                host=self.host,
                api_token=self.api_token,
                cluster_id=self.cluster_id,
                cluster_driver_port=self.cluster_driver_port,
            )
        else:
            raise ValueError(
                "Must specify either endpoint_name or cluster_id/cluster_driver_port."
            )

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "databricks-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [
            ChatMLflowAIGateway._convert_message_to_dict(message)
            for message in messages
        ]
        data: Dict[str, Any] = {
            "messages": message_dicts,
            **(self.params.dict() if self.params else {}),
        }

        resp = self._client.post(data)

        return ChatMLflowAIGateway._create_chat_result(resp)

# COMMAND ----------

from langchain.chat_models import ChatMLflowAIGateway
from langchain.schema import HumanMessage, SystemMessage

chat = ChatDatabricks(cluster_driver_port="7777")

messages = [
    SystemMessage(
        content="You are an expert in Machine Learning and Prompt Engineering specializing in helping users understand Machine Learning concepts. You have helped many people before me to gain a better understanding of Machine Learning for their projects."
    ),
    HumanMessage(
        content="What is ML?"
    ),
]
print(chat(messages))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Different cluster
# MAGIC If using a different cluster, it's required to also specify `cluster_id`, which you can find in the cluster configuration page.

# COMMAND ----------

# MAGIC %pip install -q -U langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from langchain.chat_models import ChatMLflowAIGateway
from langchain.schema import HumanMessage, SystemMessage

chat = ChatMLflowAIGateway(
    gateway_uri="http://10.68.158.191:7777",
    route="chat",
    params={
        "temperature": 0.1,
        "max_tokens": 768,
    }
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French: I love programming."
    ),
]
print(chat(messages))

# COMMAND ----------



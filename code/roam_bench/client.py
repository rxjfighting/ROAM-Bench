"""Apply the benchmark sampling and reasoning configuration to model requests."""
from .protocol import llm
from .protocol.interface_transport import client_class

ChatClient = client_class(llm.ChatClient, llm)

class EvaluationClient(ChatClient):

    def __init__(self, *args, temperature=0.0, top_p=1.0, reasoning_effort='max', **kwargs):
        super().__init__(*args, temperature=temperature, **kwargs)
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort

    def _body(self, token_scale=1):
        body = super()._body(token_scale)
        if self.top_p is not None:
            body['top_p'] = self.top_p
        if self.reasoning_effort is not None:
            body['reasoning_effort'] = self.reasoning_effort
        for name in self.dropped:
            body.pop(name, None)
        return body

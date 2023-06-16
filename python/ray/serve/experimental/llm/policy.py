from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List
from ray.serve.experimental.llm.queue import InferenceRequest, RequestQueue


class RequestSelectionPolicy(ABC):
    @abstractmethod
    def select_new_requests(
        self, in_process_requests: List[InferenceRequest], queue: RequestQueue, has_oom: bool,
    ) -> List[InferenceRequest]:
        raise NotImplementedError
    
    def request_finished(self, finished_request: InferenceRequest):
        pass

    # TODO: we might also interested in other events, such as when a request is
    # finished, or when a token is generated.


Quota = namedtuple("Quota", ["min_num_requests", "token_budget"])


class QuotaBasedRequestSelectionPolicy(RequestSelectionPolicy):
    def __init__(
        self,
        max_batch_total_tokens: int = 32000,
        waiting_served_ratio: float = 1.2,
        max_waiting_tokens: int = 20,
    ):
        self.max_batch_total_tokens = max_batch_total_tokens
        self.waiting_served_ratio = waiting_served_ratio
        self.max_waiting_tokens = max_waiting_tokens
        self.waiting_tokens = 0
        self.oom_penalty = 1.0
        self.oomed_requests = set()

    def request_finished(self, finished_request: InferenceRequest):
        if finished_request.id in self.oomed_requests:
            self.oomed_requests.remove(finished_request.id)
        if len(self.oomed_requests) == 0:
            self.oom_penalty = 1
    
    def _calculate_budget(self, requests, request):
        max_input_length = request.input_length()
        gen_length = request.gen_length()
        for r in requests:
            max_input_length = max(max_input_length, r.input_length)
            gen_length += r.gen_length

        


    def select_new_requests(
        self, in_process_requests: List[InferenceRequest], queue: RequestQueue, has_oom: bool,
    ) -> List[InferenceRequest]:
        if has_oom:
            self.oom_penalty = 0.7
            for r in in_process_requests:
                self.oomed_requests.add(r.id)
        min_num_requests, token_budget = self.calculate_quota(in_process_requests, has_oom)
        self.waiting_tokens += 1

        if min_num_requests and len(queue) < min_num_requests:
            return []

        results = []
        while not queue.empty():
            request = queue.peek()
            if request.total_tokens() >= token_budget:
                break
            results.append(request)
            queue.pop()
            token_budget -= request.total_tokens()

        if min_num_requests and len(results) < min_num_requests:
            for request in results:
                queue.reverse_push(request)
            return []

        if results:
            self.waiting_tokens = 0

        return results

    def calculate_quota(self, in_process_requests: List[InferenceRequest], has_oom) -> Quota:
        if not in_process_requests:
            return Quota(
                min_num_requests=None, token_budget=int(self.max_batch_total_tokens * self.oom_penalty)
            )

        batch_size = len(in_process_requests)

        # calculate minmal_new_requests to be served
        if self.waiting_tokens >= self.max_waiting_tokens:
            min_num_requests = 0
        else:
            min_num_requests = int(batch_size * self.waiting_served_ratio)

        # calculate token budget
        # TODO: we might want consider padding as well.
        # TODO: can we calculate the token budget based on the model?
        token_budget = max(
            0,
            int(self.max_batch_total_tokens * self.oom_penalty)
            - sum([r.total_tokens() for r in in_process_requests]),
        )
        return min_num_requests, token_budget


class StaticBatchPolicy(RequestSelectionPolicy):
    def __init__(
        self,
        batch_size: int,
    ):
        self.batch_size = batch_size

    def select_new_requests(
        self, in_process_requests: List[InferenceRequest], queue: RequestQueue, has_oom,
    ) -> List[InferenceRequest]:
        if in_process_requests:
            return []
        if len(queue) < self.batch_size:
            return []

        results = []
        while not queue.empty() and len(results) < self.batch_size:
            request = queue.peek()
            results.append(request)
            queue.pop()

        return results

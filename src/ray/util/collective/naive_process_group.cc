#include "ray/util/naive_process_group.h"

namespace c10d {
bool NaiveProcessGroup::NaiveWork::isCompleted() { return true; }

bool NaiveProcessGroup::NaiveWork::isSuccess() const { return true; }

bool NaiveProcessGroup::NaiveWork::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> NaiveProcessGroup::NaiveWork::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
NaiveProcessGroup::NaiveProcessGroup(int rank, int size) : ProcessGroup(rank, size) {}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::allgather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllgatherOptions &ops) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLGATHER,
      getNcclPG().allgather(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::_allgather_base(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllgatherOptions &ops) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLGATHER_BASE,
      getNcclPG().allgather(outputTensors, inputTensors, opts)->getFuture());
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::allreduce(
    std::vector<at::Tensor> &tensors, const AllreduceOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLREDUCE, getNcclPG().allreduce(tensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::allreduce_coalesced(
    std::vector<at::Tensor> &tensors, const AllreduceCoalescedOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLREDUCE_COALESCED,
      getNcclPG().allreduce_coalesced(tensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::alltoall(
    std::vector<at::Tensor> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllToAllOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLTOALL,
      getNcclPG().alltoall(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::alltoall_base(
    at::Tensor &outputTensor,
    at::Tensor &inputTensor,
    std::vector<int64_t> &outputSplitSizes,
    std::vector<int64_t> &inputSplitSizes,
    const AllToAllOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLTOALL_BASE,
      getNcclPG()
          .alltoall_base(
              outputTensors, inputTensors, outputSplitSizes, inputSplitSizes, opts)
          ->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::barrier(
    const BarrierOptions &opts) {
  return c10::make_intrusive<NaiveWork>(c10d::OpType::BARRIER,
                                        getNcclPG().barrier(opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::broadcast(
    std::vector<at::Tensor> &tensors, const BroadcastOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::BROADCAST, getNcclPG().broadcast(tensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::gather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const GatherOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::GATHER,
      getNcclPG().gather(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::reduce(
    std::vector<at::Tensor> &tensors, const ReduceOptions &opts) {
  return c10::make_intrusive<NaiveWork>(c10d::OpType::REDUCE,
                                        getNcclPG().reduce(tensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::reduce_scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ReduceScatterOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::REDUCE_SCATTER,
      getNcclPG().reduce_scatter(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ScatterOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::SCATTER,
      getNcclPG().scatter(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::send(
    std::vector<at::Tensor> &tensors, int dstRank, int tag) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::SEND, getNcclPG().send(tensors, dstRank, tag)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::recv(
    std::vector<at::Tensor> &tensors, int srcRank, int tag) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::RECV, getNcclPG().recv(tensors, srcRank, tag)->getFuture());
}

c10::intrusive_ptr<ProcessGroup::Work> NaiveProcessGroup::recvAnysource(
    std::vector<at::Tensor> &tensors, int tag) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::RECVANYSOURCE, getNcclPG().recvAnysource(tensors, tag)->getFuture());
}

c10::intrusive_ptr<ProcessGroup> NaiveProcessGroup::createNaiveProcessGroup(
    const c10::intrusive_ptr<::c10d::Store> & /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float> & /* unused */) {
  return c10::make_intrusive<NaiveProcessGroup>(rank, size);
}

c10d::ProcessGroupNCCL &NaiveProcessGroup::getNcclPG() {
  if (!ncclPG_) {
    ncclPG_ = c10::make_intrusive<c10d::ProcessGroupNCCL>(
        c10::make_intrusive<c10d::PrefixStore>("nccl", store_),
        getRank(),
        getSize(),
        c10d::ProcessGroupNCCL::Options::create(isHighPriorityStream_));
  }
  return *ncclPG_;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createNaiveProcessGroup", &NaiveProcessGroup::createNaiveProcessGroup);
}
}  // namespace c10d
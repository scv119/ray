#include "naive_process_group.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>

namespace py = pybind11;

namespace c10d {

namespace {
PyObject *ConvertToPythonTensor(at::Tensor &tensor) { return THPVariable_Wrap(tensor); }

void CallRayReduceAverage(std::vector<at::Tensor> &tensors) {
    py::gil_scoped_acquire acq{};
    auto module = py::module_::import("ray.util.debug");
    auto python_function = module.attr("debug_torch_tensors");
    std::vector<py::object> tensor_vectors;
    for (auto& t: tensors) {
      tensor_vectors.push_back(py::reinterpret_borrow<py::object>(ConvertToPythonTensor(t)));
    }
    python_function(py::cast(tensor_vectors));
}

void CallRayFunction() {
    py::gil_scoped_acquire acq{};
    auto module = py::module_::import("ray");
    auto python_function = module.attr("is_initialized");
    python_function();
    py::exec(
    "import ray; print(ray.is_initialized());");
    py::exec(
    "import ray; print(f'[extension] my task_id {ray.get_runtime_context().get_task_id()}')");

}
}  // namespace

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
NaiveProcessGroup::NaiveProcessGroup(const c10::intrusive_ptr<::c10d::Store> &store,
                                     int rank,
                                     int size)
    : ProcessGroup(rank, size), store_(store) {}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> NaiveProcessGroup::allgather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllgatherOptions &ops) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLGATHER,
      getNcclPG().allgather(outputTensors, inputTensors, ops)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::_allgather_base(
    at::Tensor &outputBuffer, at::Tensor &inputBuffer, const AllgatherOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::_ALLGATHER_BASE,
      getNcclPG()._allgather_base(outputBuffer, inputBuffer, opts)->getFuture());
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> NaiveProcessGroup::allreduce(std::vector<at::Tensor> &tensors,
                                                      const AllreduceOptions &opts) {
  CallRayFunction();
  CallRayReduceAverage(tensors);
//  if (opts.reduceOp != ReduceOp::AVG) {
    return c10::make_intrusive<NaiveWork>(
        c10d::OpType::ALLREDUCE, getNcclPG().allreduce(tensors, opts)->getFuture());
//  }

//  CallRayReduceAverage(tensors);
//
//  auto future = c10::make_intrusive<c10::ivalue::Future>(
//    c10::ListType::create(c10::TensorType::get()));
//  future->markCompleted(c10::IValue(tensors));
//  return c10::make_intrusive<NaiveWork>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> NaiveProcessGroup::allreduce_coalesced(
    std::vector<at::Tensor> &tensors, const AllreduceCoalescedOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLREDUCE_COALESCED,
      getNcclPG().allreduce_coalesced(tensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::alltoall(
    std::vector<at::Tensor> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllToAllOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLTOALL,
      getNcclPG().alltoall(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::alltoall_base(
    at::Tensor &outputTensor,
    at::Tensor &inputTensor,
    std::vector<int64_t> &outputSplitSizes,
    std::vector<int64_t> &inputSplitSizes,
    const AllToAllOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::ALLTOALL_BASE,
      getNcclPG()
          .alltoall_base(
              outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts)
          ->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::barrier(const BarrierOptions &opts) {
  return c10::make_intrusive<NaiveWork>(c10d::OpType::BARRIER,
                                        getNcclPG().barrier(opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::broadcast(std::vector<at::Tensor> &tensors,
                                                      const BroadcastOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::BROADCAST, getNcclPG().broadcast(tensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::gather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const GatherOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::GATHER,
      getNcclPG().gather(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::reduce(std::vector<at::Tensor> &tensors,
                                                   const ReduceOptions &opts) {
  return c10::make_intrusive<NaiveWork>(c10d::OpType::REDUCE,
                                        getNcclPG().reduce(tensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::reduce_scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ReduceScatterOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::REDUCE_SCATTER,
      getNcclPG().reduce_scatter(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ScatterOptions &opts) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::SCATTER,
      getNcclPG().scatter(outputTensors, inputTensors, opts)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::send(std::vector<at::Tensor> &tensors,
                                                 int dstRank,
                                                 int tag) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::SEND, getNcclPG().send(tensors, dstRank, tag)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::recv(std::vector<at::Tensor> &tensors,
                                                 int srcRank,
                                                 int tag) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::RECV, getNcclPG().recv(tensors, srcRank, tag)->getFuture());
}

c10::intrusive_ptr<Work> NaiveProcessGroup::recvAnysource(
    std::vector<at::Tensor> &tensors, int tag) {
  return c10::make_intrusive<NaiveWork>(
      c10d::OpType::RECVANYSOURCE, getNcclPG().recvAnysource(tensors, tag)->getFuture());
}

c10::intrusive_ptr<ProcessGroup> NaiveProcessGroup::createNaiveProcessGroup(
    const c10::intrusive_ptr<::c10d::Store> &store,
    int rank,
    int size,
    const std::chrono::duration<float> & /* unused */) {
  return c10::make_intrusive<NaiveProcessGroup>(store, rank, size);
}

c10d::ProcessGroup &NaiveProcessGroup::getNcclPG() {
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

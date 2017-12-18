#include <vector>
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");

  // LOG(INFO) << "forward gpu: write_count is: ";
  // int offset = batch->label_.count()/8;
  //   for(int i=0;i<8;i++)
  //     LOG(INFO) << batch->label_.cpu_data()[0+offset*i] << "," << batch->label_.cpu_data()[1+offset*i] << ","
  //               << batch->label_.cpu_data()[2+offset*i];

  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());

  // offset = batch->data_.count()/8;
  // for(int i=0;i<8;i++)
  //   LOG(INFO) << batch->data_.cpu_data()[0+offset*i] << "," << batch->data_.cpu_data()[1+offset*i] << ","
  //             << batch->data_.cpu_data()[2+offset*i];

  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe

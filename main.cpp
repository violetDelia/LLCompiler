// #include "onnx/common/file_utils.h"
#include "llcompiler/utils/logger.h"
#include <fstream>
#include <iostream>
#include <string>
#include <memory>


int main() {
  llc::logger::register_logger("c","");
  llc::logger::register_logger("file",".");
  // LLCOMPILER_REGISTER_LOGGER("default", "");
  // LLCOMPILER_REGISTER_LOGGER("file", ".");

  // std::string fname =
  //     "E:/Tianyi_sync_folder/code/LLCompiler/models/resnet18-v1-7.onnx";
  // onnx::ModelProto model;
  // LoadProtoFromPath(fname, model);
  // std::cout<<typeid(model).name()<<std::endl;
  // std::cout<<model.ByteSize()<<std::endl;
  // std::cout<<model.GetDescriptor()->name()<<std::endl;
  // std::cout<<model.GetDescriptor()->full_name()<<std::endl;
  // std::cout<<model.GetDescriptor()->index()<<std::endl;
  // std::cout<<model.GetDescriptor()->DebugString()<<std::endl;
  // std::cout<<model.GetDescriptor()->field_count()<<std::endl;
  // std::string model_string;
  // std::cout<<model.SerializeToString(&model_string)<<std::endl;;
  // std::cout << model_string;
}
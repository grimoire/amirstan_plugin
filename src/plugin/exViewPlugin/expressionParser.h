#pragma once
#include <string>
#include "NvInferPlugin.h"

namespace amirstan
{
namespace plugin
{

const nvinfer1::IDimensionExpr* parse_expression(const std::string& exp, const nvinfer1::DimsExprs *inputs, nvinfer1::IExprBuilder &exprBuilder);

}
}
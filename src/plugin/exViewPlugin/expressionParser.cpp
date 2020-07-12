#include "expressionParser.h"
#include <ctype.h>
#include <iostream>

namespace amirstan
{
namespace plugin
{
// return if the symbol is number
inline bool next_symbol(const std::string& exp, int start_pos, int& next_pos, int& symbol){
    
    // end of string
    if(start_pos>=exp.size()){
        next_pos = start_pos;
        symbol = -1;
        return false;
    } 

    next_pos = start_pos+1;
    // not number
    if(exp[start_pos]<'0' || exp[start_pos]>'9'){
        symbol = int(exp[start_pos]);
        return false;
    }

    // number
    symbol = int(exp[start_pos]-'0');
    while(
        next_pos<exp.size() && 
        (exp[next_pos]<='9' && exp[next_pos]>='0')
    ){
        symbol = symbol*10 + int(exp[next_pos]-'0');
        next_pos += 1;
    }
    
    return true;
}

const nvinfer1::IDimensionExpr* get_value_impl(const std::string& exp, const nvinfer1::DimsExprs *inputs, int start_pos, int& next_pos, nvinfer1::IExprBuilder &exprBuilder);
const nvinfer1::IDimensionExpr* parse_expression_impl(const std::string& exp, const nvinfer1::DimsExprs *inputs, int start_pos, int& next_pos, nvinfer1::IExprBuilder &exprBuilder);

const nvinfer1::IDimensionExpr* get_value_impl(const std::string& exp, const nvinfer1::DimsExprs *inputs, 
int start_pos, int& next_pos, nvinfer1::IExprBuilder &exprBuilder){
    if(start_pos >= exp.size()){
        return nullptr;
    }
    next_pos = start_pos;
    int symbol = 0;
    bool isnumber = next_symbol(exp, next_pos, next_pos, symbol);

    // get init value
    const nvinfer1::IDimensionExpr* result = nullptr;
    if(isnumber){
        result = exprBuilder.constant(symbol);
    }else{
        if(isalpha(symbol)){
            int desc_id = tolower(symbol)-'a';
            isnumber = next_symbol(exp, next_pos, next_pos, symbol);
            if(!isnumber){
                std::cout << "wrong expression: " << exp << std::endl;
                return nullptr;
            }
            result = inputs[desc_id].d[symbol];
        }else if(char(symbol) == '('){
            result = parse_expression_impl(exp, inputs, start_pos+1, next_pos, exprBuilder);
            if(next_pos>=exp.size() || exp[next_pos]!=')'){
                next_pos+=1;
                std::cout << "wrong expression: " << exp << std::endl;
                return nullptr;
            }
            next_pos+=1;
            
        }else{
            std::cout << "wrong expression: " << exp << std::endl;
            return nullptr;
        }
    }

    return result;
}

const nvinfer1::IDimensionExpr* parse_expression_impl(const std::string& exp, const nvinfer1::DimsExprs *inputs,
 int start_pos, int& next_pos, nvinfer1::IExprBuilder &exprBuilder){

    if(start_pos >= exp.size()){
        return nullptr;
    }
    next_pos = start_pos;
    int symbol = 0;
    bool isnumber = false;

    // get init value
    auto return_value = get_value_impl(exp, inputs, next_pos, next_pos, exprBuilder);

    // loop
    for(int i=next_pos;i<exp.size();++i){
        isnumber = next_symbol(exp, next_pos, next_pos, symbol);
        char chr_sym = char(symbol);

        if(chr_sym==')'){
            next_pos-=1;
            break;
        }

        auto result = get_value_impl(exp, inputs, next_pos, next_pos, exprBuilder);
        
        switch (chr_sym){
        case '+':
            return_value = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, 
                *return_value,
                *result);
            break;
        case '-':
            return_value = exprBuilder.operation(nvinfer1::DimensionOperation::kSUB, 
                *return_value,
                *result);
            break;
        case '*':
            return_value = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, 
                *return_value,
                *result);
            break;
        case '/':
            return_value = exprBuilder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV, 
                *return_value,
                *result);
            break;
        default:
            break;
        }
    }
    return return_value;
}
    
const nvinfer1::IDimensionExpr* parse_expression(const std::string& exp, const nvinfer1::DimsExprs *inputs, nvinfer1::IExprBuilder &exprBuilder){
    int end_pos;
    return parse_expression_impl(exp, inputs, 0, end_pos, exprBuilder);
}
}
}
#pragma once
#include <array>
#include "Parameter.hpp"


enum class Operation
{
	LEAF,
	//these are unary
	RELU,

	// these are non unary
	ADD,
	SUBSTRACT,
	MULTIPLY,
	DIVIDE,
	SQUARE



};

struct Operation
{
	virtual void forward();
	virtual void backward();
};

// node should be at most binary. 
template <typename T>
struct Node
{
	Node(Operation inputOp) : op(inputOp) {};
	~Node() = default;
	Parameter<T> param{};
	std::array<Node<T>*, 2> children = { nullptr, nullptr };
	Operation op{ Operation::LEAF };
};

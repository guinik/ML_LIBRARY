
#pragma once
#include <array>

template <typename F>
struct Parameter{
	Parameter() = default;
	~Parameter() = default;

	F value;
	F grad;

	void clearGradients() {
		grad = value;
		grad.fillValues(0.0f);
	};
};

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

// node should be at most binary. 
template <typename T>
struct Node
{
	Node(Operation inputOp) : op(inputOp) {};
	~Node() = default;
	Parameter<T> param{};
	std::array<Node<T>*, 2> children = { nullptr, nullptr };
	Operation op{Operation::LEAF};
	bool requires_grad = true;
};
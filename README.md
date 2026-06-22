# Linear Models for Predicting PUF Behavior

A machine learning based security research project focused on modeling and predicting the behavior of **Physical Unclonable Functions (PUFs)** using linear models, kernel methods, and optimization techniques.

This project explores the vulnerability of hardware security primitives by deriving mathematical models capable of learning and predicting PUF responses with high accuracy.

---

## Project Overview

Physical Unclonable Functions (PUFs) are hardware security primitives widely used for authentication and cryptographic applications.

This project investigates whether machine learning algorithms can successfully model and predict the behavior of:

- Arbiter PUF
- Multi-Level XOR PUF (ML-PUF)

The goal was to derive mathematical representations of delay behavior and evaluate whether learning algorithms can recover hidden circuit characteristics.

---

## Objectives

- Derive a linear mathematical model for predicting Arbiter PUF behavior
- Extend the model to Multi-Level XOR PUF architecture
- Reduce high-dimensional feature space using mathematical simplification
- Design a kernel function for SVM-based classification
- Recover internal circuit delay parameters through inverse optimization
- Evaluate performance of different machine learning classifiers

---

## Methodology

### 1. Linear Modeling of Arbiter PUF

Derived recursive equations for signal propagation delay across multiple switching stages.

The final model was expressed as:

WᵀΦ(C) + b

allowing PUF behavior to be represented as a linear classification problem.

---

### 2. Feature Engineering

Constructed high-dimensional feature mapping for XOR-PUF systems.

- Initial feature dimension: 256
- Reduced dimension: 87 features

Reduction was achieved using mathematical simplification and elimination of redundant feature combinations.

---

### 3. Kernel Design

Designed Polynomial Kernel for high-dimensional feature learning.

Kernel Function:

K(C,C’) = (1 + D·D’)⁸

Parameters:

- Degree = 8
- Gamma = 1
- Coef0 = 1

---

### 4. Delay Recovery (Inverse Problem)

Recovered internal circuit delays by solving an underdetermined optimization problem.

Optimization Objective:

min ||Ax - y||²

Subject to:

x ≥ 0

Implemented using constrained Linear Regression.

---

### 5. Model Training & Evaluation

Compared performance of:

- Logistic Regression
- Linear SVM (LinearSVC)

Experimented with:

- L1 Regularization
- L2 Regularization
- Different convergence tolerances
- Different regularization strengths (C)

---

## Results

### Logistic Regression

| Penalty | Accuracy |
|----------|----------|
| L2 (C=1) | 100% |
| L1 (C=1) | 100% |

### Linear SVM

| Penalty | Accuracy |
|----------|----------|
| L2 | 100% |
| L1 | 100% |

Key Observation:

- L2 regularization consistently trained faster
- Low regularization caused underfitting
- Higher values of C provided no additional accuracy gain

---

## Tech Stack

- Python
- NumPy
- Scikit-learn
- Linear Regression
- Logistic Regression
- Support Vector Machines
- Mathematical Optimization
- Feature Engineering

---

## Concepts Used

- Machine Learning
- Hardware Security
- Physical Unclonable Functions (PUFs)
- Linear Algebra
- Kernel Methods
- Convex Optimization
- Classification Algorithms
- Regularization Techniques

---

## Repository Structure

├── notebooks/  
├── experiments/  
├── dataset/  
├── model_training.py  
├── svm_kernel.py  
├── delay_recovery.py  
├── report.pdf  
└── README.md  

---

## Key Learning Outcomes

- Learned mathematical modeling of hardware security systems
- Implemented custom feature transformation pipelines
- Understood kernel-based classification methods
- Applied constrained optimization to inverse engineering problems
- Evaluated trade-offs between accuracy and computational efficiency

---

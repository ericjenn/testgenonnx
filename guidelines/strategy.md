---
created:
  - 2026-04-29 07:44
tags:
  - note
title:
reference:
url:
status:
---
```table-of-contents
title: **Table of contents**
hideWhenEmpty: true
```
## Introduction
This document proposes an approach to develop the test inputs of the SONNX operators. It does not (yet) cover the actual implementation of tests (using [Hypothesis](https://hypothesis.works/) or any other test framework such as [pytest](https://docs.pytest.org/en/stable/), [doctest](https://docs.python.org/3/library/doctest.html), etc.). This will be done in a future version of the document.

The current version of this note is organized as follows:
- The first section recalls the objectives of testing in the context of SONNX and give some elements about the test strategy,
- The second section focuses on the practical implementation of the test strategy for SONNX operators

### Test objectives and scope in SONNX

In SONNX, tests target two main objectives:
- **Validating** the informal specification through comparison with existing implementations (e.g., ONNX runtime)
- **Verifying** implementations of the SONNX specification, using the SONNX reference implementation as the test oracle.

Concerning the first objective, it could be interpreted at first sight as some sort of "retro-engineering" activity in which the specification is derived from the implementation. *This is clearly not the case*. Tests against an implementation is a way to increase our confidence in the SONNX specification by comparing it to existing reliable implementations.
In case of discrepancy, no immediate conclusion can be drawn since the error may be either in the specification or in the implementation. However, this reveals a problem in either or both sides that needs to be addressed. This approach has already  been fruitful by raising issues both in our specification and in the ONNX runtime implementation.

Concerning the second objective, it is worth noting that passing all tests developed in SONNX does not **guarantee** full  conformance with the SONNX specification. In particular, in the context of the development of a certified system, demonstration of conformance remains the responsibility of the applicant.

#### Note on scope: input equivalence vs. output-comparison tolerance

Everything from § Equivalence class-based testing onward partitions the **input** space of an operator. The first objective above — validating the informal specification against an existing implementation such as ONNX runtime — also requires a notion of equivalence on the **output** side: a rule for deciding whether two floating-point results (the SONNX reference output and the runtime-under-test output) count as "the same," typically an ULP-based or relative-error tolerance. This output-comparison tolerance is a separate, orthogonal test parameter, not a byproduct of how the input classes were chosen, and it should be specified once per data type, independently of the input equivalence-class machinery described below. Otherwise it is easy to conflate "these two inputs are equivalent for testing purposes" with "these two outputs are close enough to call equal" — different relations serving different purposes (test *generation* vs. test *oracle comparison*).

### Test strategies
#### Functional *vs. implementation-based* tests

In SONNX, tests are essentially *functional*, i.e., they only refer to *what* the operators are expected to do. They do not consider *how* they will be implemented.

However, in practice, additional tests may be defined to account for possible implementations solutions and well-known sources of errors. In that case, the reason for the test shall be clearly documented.

#### Test traceability

Tests must be traceable to a specific part of the informal specification. Traceability shall be done using reference to the specification section or to a specic requirement tag (e.g., `E_DIV_REAL_FUNC_010`).

Where a test is generated from an equivalence class rather than written by hand, this traceability tag is recorded alongside the class-provenance metadata introduced in § Eliminating irrelevant cases (which criterion, or which unified pair of criteria, generated the test, and whether the specific combination was retained because of a stated fault model or because it filled a pairwise-coverage obligation). A test case is then fully self-documenting: which specification requirement it targets, which class(es) it instantiates, and — if applicable — which named cross-criterion interaction motivated it.

#### Tets inputs
An ONNX operator has 0, 1 or several arguments,  0, 1, or several attributes. Arguments and attributes are tensors (possibly scalars).

A tensor $T$ has $n$-dimensions (or "has $n$ axes" or "is of rank-$n$") with $n\ge 0$.   An element of the tensor is noted  $T[i_0,i_1,\dots,i_{n-1}]$ where $i_k$ is the index along axis $k$. We have $i_k \in [0,dT_k]$ where $dT_k$ is the number of elements along the $k$-th axis.

So, the input of an operator with a unique argument $T$ is the vector of values  $({dT_1, dT_2,...,dT_{n-1},T[i]~\text{for all valid index}~i})$.

From the perspective of the test, each element of this vector is actually a specific dimension of the test space.

#### Equivalence class-based testing

Except for very simple cases such as e.g., **Add** on int8 scalars, it is generally impossible test an operator against all its possible input values. Therefore, tests are generally performed on a selected subset of all possible input values.

One strategy to select those values can be based on the concept of *equivalence classes*.

Considering an equivalence relation $R$ (i.e., a relation that is reflexive, symmetric, and transitive), the equivalence class $C$ of an element $x$ is the maximal set of values equivalent to $x$ under $R$: $C=\{y \mid xRy\}$. Two values $x$ and $y$ belong to the same class iff $xRy$; consequently, for any $a,b \in C$, $aRb$ holds.

In the context of testing, the relation $R$ is related to the capability of the test vector to reveal some design or implementation error. So, if $a R b$,  $a$ and $b$ are in the same class, then an error revealed by input $a$ is also revealed by input $b$. Stated differently, input $a$ is equivalent to $b$ from from the perspective of testing.

For instance, testing **Add(x1,x2)** with $a=(x_1:10,x_2:20)$ is considered to reveal the same errors as with $b=(x_1:15, x_2:-10)$.

Obviously, the actual capability of the input to activate and propage a fault depends one the fault model -- which is unknown --, and the part of the design that is exercised, which is also unknown at specification level. So, a weaker definition is to consider that $R$ is related to the "coverage" of the functional behaviour of the operator exercised by the test inputs.

For instance, we may define the classes as follows: two inputs $a$ and $b$ belong to the same class if
- $a$ and $b$ are on the same boundary of the input domain, and
- $a$ and $b$ exercise the same branch for a specification involving multiple branches (or conditions), and
- $a$ and $b$ uses the same special values, and
- if an operator involves an other operators **Op**, $a$ and $b$ are in the same class with respect to **Op**.

The actual definition is a bit more complicated, let's refine it.
#### Equivalence classes based on the input domain

##### Principles

 A first contribution to the definition of equivalence classes is the domain of validity of values of the inputs.

This domain is defined by
- the constraint determined by the types of the inputs (e.g., $x \in [0,255]$ if $x$ is an unsigned integer),
- all constraints concerning the inputs defined in the specification (e.g., $dY_2 = \left\lfloor{((dX_2 + pad\_shape[0] - (\texttt{dilations}[0] * (\texttt{kernel\_shape}[0] - 1) + 1)) / \texttt{strides}[0]) + 1}\right\rfloor$ for the **MaxPool operator**)

Once the domain is defined, we discriminate
- the class of all values located *on the boundary of the domain*,
- the class of all values *inside the domain defined by the boundary* (the boundary themselves being excluded). The domain may be composed of disjoint parts, with each part defining its own class.

If the domain is determined by multiple constraints, we define a class for each combination of constraints satisfied simultaneously.

The picture below gives a very simple example of a domain with two variables whose constraints are independent. In the general case, the domain will involve many more variables and more complex boundaries (consider the constraint given earlier about `dY2`).
![[file-20260911092415027.png|371]]

Let us consider a domain defined by some predicate $p_i(x_1,x_2,...,x_n)$ over a subset of the operator inputs. A predicate expresses a relation (an inequality or an equality) involving one or several inputs.
For instance:
- $x_1< 10$
- $x_1+x_2\leq x_3$
- $x_1^2+x_2^2=c$ (a circular domain)
- etc.

The domain is defined by the set of predicates $P={p_1,p_2,...,p_m}$.

We want to distinguish the points located on the boundary of the domain defined by the set of constraints from the points located strictly in the domain (i.e, not on the boundary).

First, we replace all strict inequality by non-strict inequality.
For instance, in the integer domain, predicate $x_1<10$ become $x_1\le 9$.
In the floating point domain, $x_1+x_2<x_3$ becomes $x_1+x_2 \le \text{nextdown}(x_3)$ with $\text{nextdown}(x_3)$ being the largest floating point number strictly smaller than $x_3$, considering the round-to-nearest, ties-to-even IEEE754 rounding mode.

Then,  we decompose each (possibly rewritten) inequality predicate $p_i$ into two predicates $p'_i$ and $p''_i$ such that
- $p'i$ is obtained by replacing operator $\le$ and $\ge$  in $p_i$ by operator  $=$
- $p''_i$ is obtained by  replacing operators $\le$ (resp.  $\ge$)  in $p_i$ by operator $\lt$ (resp. $\gt$).
and equality predicates $p_i$ are kept as is and denoted $p'_i$.

The initial system is $p_1 \wedge p_2 \wedge ... \wedge p_n = (p'_1 \vee p''_1) \wedge (p'_2 \vee p''2) \wedge ... \wedge (p'_n \vee p''_n)$ , which can be written  $(p'_1 \wedge p'_2 \wedge ... \wedge p'_n) \vee (p'_1 \wedge p'_2 \wedge ... \wedge p'_{n-1} \wedge p''_n) \vee ... \vee (p''_1 \wedge p''_2 \wedge... \wedge p''_n)$ $

Note that, by construction, $p'_i$ and $p''_i$ are disjoint so all combinations involving the product $p'i \wedge p''_i$ can  be simply ignored since they cannot be satisfied. However, to simplify the description, we keep them.

Each member of the last expression related by the $\vee$ operator represents a sub-domain. The first sub-domain $p'_1 \wedge p'_2 \wedge ... \wedge p'_n$  is the most constrained since all $p'_i$ are equality relations (in the previous figure, it corresponds to one of the 4 corners). Conversely, the last one is the least constrained.

Each member represent an equivalence relation, hence a testing domain.

**Note.** Because this expansion is exhaustive by construction, it already *is* the full combinatorial enumeration of every joint boundary/interior choice across all predicates in $P$ — nothing beyond it is needed to combine classes defined over a shared predicate set. The only place where a separate mechanism is required is where two classes come from predicate sets that are *not yet* shared, e.g. one built from the operator's arithmetic pre-conditions and another built from operator-specific special-value knowledge; see § Eliminating irrelevant cases for how such classes are unified and, once unified, reduced to a tractable set.

Let us take for example the following set of constraints, with $x_i \in \mathbb{N}$:
- $p_1(x_1) \Leftrightarrow x_1 < 100$
- $p_2(x_1,x_2) \Leftrightarrow x_1+x_2 < 150$

The predicate $p_i(x_1)\Leftrightarrow (x_1 < 100) \Leftrightarrow (x_1 \le 99)$ gives
- $p'_i(x_1) \Leftrightarrow x_1 = 99$ and
- $p''_i(x_1) \Leftrightarrow x_1 \lt 99$.

The predicate $p_2(x_1,x_2) \Leftrightarrow (x_1+x_2 < 150)  \Leftrightarrow (x_1+x_2 \le 149)$  gives
- $p'_2(x_1,x_2) \Leftrightarrow x_1+x_2 = 149$ and
- $p''_2(x_1,x_2) \Leftrightarrow x_1+x_2 \lt 149$.

This system of constraints will lead to the following domains:
1. $p'1 \wedge p'_2 \Leftrightarrow (x_1=99) \wedge (x_1+x_2=149)$
2. $p'1_1 \wedge p''_2 \Leftrightarrow (x_1 = 99) \wedge (x_1+x_2 \lt 149)$
3. $p''_1 \wedge p'_2 \Leftrightarrow (x_1\le 98) \wedge (x_1+x_2=149)$
4. $p''_1 \wedge p''_2 \Leftrightarrow (x_1<99) \wedge (x_1+x_2<149)$

Now, let's consider the simple case of the **Add(x1,x2)** for `unsigned int8` values.
Variable $x_1$ is in the domain $[0,255]$, variable $x_2$ is in the domain $[0,255]$.

The system of equations is:
- $p_1 \Leftrightarrow x_1\ge 0$
- $p_2 \Leftrightarrow x_1\le 255$
- $p_3 \Leftrightarrow x_2 \ge 0$
- $p_4 \Leftrightarrow x_2 \le 255$

(Note that we may also consider the constraint about the type of the output, e.g., $x_1+x_2 \leq 255$ for unsigned ints.)

This lead to the follow systems of equations:
- 4 out of 4 combination of $p'_i$:
	- $P_{4/4} \Leftrightarrow p'1 \wedge p'_2 \wedge p'_3 \wedge p'_4$
- 3 out of 4 combinations of $p'_i$:
	- $P_{3/4.1} \Leftrightarrow p'1 \wedge p'_2 \wedge p'_3 \wedge p''_4$
	- $P_{3/4.2} \Leftrightarrow p'1 \wedge p'_2 \wedge p''_3 \wedge p'_4$
	- $P_{3/4.3} \Leftrightarrow p'1 \wedge p''_2 \wedge p'_3 \wedge p'_4$
	- $P_{3/4.4} \Leftrightarrow  p''1 \wedge p'_2 \wedge p'_3 \wedge p_4$
- 2 out of 4 combinations of $p'_i$:
	- $P_{2/4.1} \Leftrightarrow  p'1 \wedge p'_2 \wedge p''_3 \wedge p''_4$
	- ...
	- $P_{2/4.6} \Leftrightarrow  p''1 \wedge p''_2 \wedge p'_3 \wedge p'_4$
- 1 out of 4 combinations of $p'_i$:
	- $P_{1/4.1} \Leftrightarrow  p'1 \wedge p''_2 \wedge p''_3 \wedge p''_4$
	- $P_{1/4.2} \Leftrightarrow p''1 \wedge p'_2 \wedge p''_3 \wedge p''_4$
	- $P_{1/4.3} \Leftrightarrow p''1 \wedge p''_2 \wedge p'_3 \wedge p''_4$
	- $P_{1/4.4} \Leftrightarrow p''1 \wedge p''_2 \wedge p''_3 \wedge p'_4$
- no $p'_i$:
	- $P_{0/4} \Leftrightarrow  p''1 \wedge p''_2 \wedge p''_3 \wedge p''_4$

Applied to our example, the problem simplifies since some of the predicates are incompatible (for instance $p'1$ and $p'_2$ cannot be true simultaneously and $p'_1 \implies p''_1$ because they respectively define the left and right bounds of the domain). This leads to the following predicates:
- $P_{4/4}:$ *no solution*
- $P_{3/4.i, i=1..4}:$ *no solution*
- $P_{2/4.1}: p'1 \wedge p'_2 \wedge p''_3 \wedge p''_4$ : *no solution*
- $P_{2/4.2}: p'1 \wedge p''_2 \wedge p'_3 \wedge p''_4$ : $(x_1=0) \wedge (x_2=0)$
- $P_{2/4.3}: p'1 \wedge p''_2 \wedge p''_3 \wedge p'_4$ : $(x_1=0) \wedge (x_2=255)$
- $P_{2/4.4}: p''1 \wedge p'_2 \wedge p'_3 \wedge p''_4$ : $(x_1=255) \wedge (x_2=0)$
- $P_{2/4.5}: p''1 \wedge p'_2 \wedge p''_3 \wedge p'_4$ : $(x_1=255) \wedge (x_2=255)$
- $P_{2/4.6}: p''1 \wedge p''_2 \wedge p'_3 \wedge p'_4$ : *no solution*
- $P_{1/4.1}:p'_1  \wedge p''_2 \wedge p''_3 \wedge p''_4$ : $(x_1=0) \wedge (x_2\in ]0,255[)$
- $P_{1/4.2}:p''_1  \wedge p'_2 \wedge p''_3 \wedge p''_4$ : $(x_1=255)\wedge (x_2\in ]0,255[)$
- $P_{1/4.3}:p''_1 \wedge p''_2 \wedge p'_3 \wedge p''_4$ : $(x_1\in ]0,255[) \wedge (x_2=0)$
- $P_{1/4.4}:p''_1 \wedge p''_2 \wedge p''_3 \wedge p'_4$ : $(x_1\in ]0,255[) \wedge (x_2=255)$
- $P0/4: p''_1 \wedge p''_2 \wedge p''_3 \wedge p''_4$ : $(x_1\in ]0,255[) \wedge (x_2\in]0, 255[)$

The application of this strategy leads to consider 9 classes defined as follows:
- 4 classes corresponding to the 4 edges of the domain
- 4 classes corresponding to the 4 vertices of the domain (edges excluded)
- 1 class corresponding to the rest of the domain

This is illustrated on the following figure. The points represented by a red crosses correspond to 1 out of 4 combinations ; the points represented by blue crosses correspond to 2 out of 4 combinations.

![[file-20260518180058243.png|408]]

In the case of the $Div(x,y)$ for signed integers, the domain of $y$ is $]-128,0[ \cup ]0, 127]$. By applying the same strategy, we end up with the following classes:
- Edges: (-128,-128),(-128,0),(-128,127),(127,-128),(127,0),(127,127)
- Boundaries:
	- $(x=-128) \wedge (y \in ]-128,0[)$
	- $(x=-128) \wedge (y \in ]0,127[)$
	- $(x=127) \wedge (y \in ]-128,0[)$
	- $(x=127) \wedge (y \in ]0,127[)$
	- $(x\in]-127,128[) \wedge (y=-128)$
	- $(x\in]-127,128[) \wedge (y=127)$
- and the rest of the domain:
	- $(x\in]-127,128[) \wedge (y \in ]0,127[)$

In the case of SONNX, the definition of the input domain can be much more complex. and involve not only  type constraints for each input, but also constraints relating multiple inputs.

For instance, in the case of the **Maxpool** operator, the input domain is defined by the following set of constraints (we consider the restricted version of the operator specified in SONNX and we only consider the structural parameters of the tensors, not their values):
- Type constraints
	- $\text{dilations}[0..1] \in \mathbb{N}$
	- $\text{strides}[0..1] \in \mathbb{N}$
	- $\text{pads}[0..3] \in \mathbb{N}$
	- $dX_{0..3} \in \mathbb{N}$
	- $dY_{0..3} \in \mathbb{N}$
	- $dW_{0..1} \in \mathbb{N}$
	- $X[i,j,k,l] \in \mathbb{N}$
	- $Y[i,j,k,l] \in \mathbb{N}$
- Functional constraints
	- $\text{dilations}[0] > 0$
	- $\text{dilations}[1] > 0$
	- $\text{strides}[0]>0$
	- $\text{strides}[1] > 0$
	- The kernel shall not completely fit in the padding area (\*)
		- $\text{dilations}[0]\times(dW_0-1)+1 > \text{pads}[0]$
		- $\text{dilations}[0]\times(dW_0-1)+1 > \text{pads}[2]$
		- $\text{dilations}[1]\times(dW_1-1)+1 > \text{pads}[1]$
		- $\text{dilations}[1]\times(dW_1-1)+1 > \text{pads}[3]$
	- Size of the output tensor (\*\*)
		- $(dY_2-1)\times\text{strides}[0]+(\text{dilations}[0] (dW_0-1)+1) \le dX2+\text{pads}[0]+\text{pads}[2]$
		- $dY_2\times\text{strides}[0]+(\text{dilations}[0] (dW_0-1)+1) \gt dX2+\text{pads}[0]+\text{pads}[2]$

(\*) This constraint is due to the fact that the operator also returns the index of the maximum value **in the input tensor**. This index does not makes sense if the kernel completely "fits" in the padding area since, in that case, the maximum value is found in the padded area.

In ONNX runtime, the constraint is stronger: the padding shall be smaller than the kernel size (before dilation).

(\*\*) If the size of output tensor is $dY_2$ , it means that the kernel has been applied $dY_2$ times on the input. The application of the kernel shall not overflow the padded tensor, so $$(dY_2-1)\times\text{strides}[0]+(\text{dilations}[0] (dW_0-1)+1) \le dX2+\text{pads}[0]+\text{pads}[2]$$ But the kernel shall be applied as many times as possible, so we have also
$$dY_2\times\text{strides}[0]+(\text{dilations}[0] (dW_0-1)+1) \gt dX2+\text{pads}[0]+\text{pads}[2]$$

##### Selecting a representative value within a class

The decomposition above identifies *classes* — regions of the input space — not test vectors. Each class must still be reduced to one concrete input before it becomes a test, and this document previously left that step implicit. Two rules apply:

1. **Boundary classes ($p'_i$) fix their value by construction.** A class defined by $x_1=99$ has exactly one boundary value for $x_1$ along that axis, so no choice is involved there. Where a class fixes several variables jointly through an equality (e.g., $x_1+x_2=149$), any pair satisfying it is an equally valid representative *for that axis alone*; the actual pair used should be driven by whichever other axis (type-specific domain, special value, structural class) is being exercised at the same time — see § Eliminating irrelevant cases for how classes from different axes are unified — rather than picked arbitrarily.
2. **Interior classes ($p''_i$, an open interval or region) require an explicit choice rule**, because "any value in the interior" is not itself a test input — and, for floating point especially, different interior choices exercise different behaviour (a generic mid-range value, a subnormal, a value adjacent to a boundary are all "interior" but not equivalent for testing purposes). The default rule is:
	- for integer domains, use the arithmetic midpoint of the interval (rounded), unless a type-specific or special-value class from § Domains also applies to that variable, in which case that class's value is used instead;
	- for floating-point domains, the interior representative is always drawn *from* the type-specific domain classes (§ Type-specific domains: normal, subnormal, etc.) rather than from an arbitrary "generic" value. In other words, the interior of a predicate-derived class and the type-specific classes are not two independent things to be covered separately; the interior class is *instantiated by* whichever type-specific class has been unified into the same predicate set for that variable (§ Eliminating irrelevant cases).

This removes the ambiguity of a class such as $x_1=0 \wedge x_2\in\,]0,255[$ (the example above) leaving the choice of $x_2$ unstated: under this rule, $x_2$ is set by whichever type/special-value class has been unified with this boundary class, and that unification is itself recorded per § Eliminating irrelevant cases.

#####  The case of unbounded variables

All variables are bounded by their types but for some of them — typically variables describing a structural property such as a rank or a dimension — the bound defined by the type cannot be exercised in practice. For instance the right bound of the tensor rank and size domains can be -- theoretically -- as large as $2^{64}-1$, but, in reality, those bounds are impossible to reach due to physical limits on memory and/or execution time.

In that case, an explicit **usage-domain predicate** is added to the constraint set $P$ for the variable concerned, e.g., for a dimension size $dX_k$:
- $p_{lim}(dX_k) \Leftrightarrow dX_k \le 1000$

This is an ordinary predicate, not a special case: it is rewritten and split into $p'_{lim}$ ($dX_k=1000$) and $p''_{lim}$ ($dX_k<1000$) using exactly the procedure of the previous sections, and it takes part in the same combinatorial enumeration and pruning as every type- or specification-derived constraint. Nothing in the decomposition machinery changes; only the *origin* of the predicate differs — the bound reflects an accepted usage domain rather than a functional limit of the operator.

Because this bound is an engineering choice rather than a specification fact, the chosen value (1000 above, or e.g. 4 for a rank) must be:
- stated explicitly, alongside the operator's other constraints, rather than left implicit in a test generator's configuration; and
- justified using the same "fault model assumed absent" reasoning as pruning (§ Eliminating irrelevant cases) — e.g., "no known implementation pattern distinguishes a size of 1000 from a size of 10000; both exercise the same loop and indexing logic" — so that a reviewer can challenge the choice of limit itself, and not only the classes subsequently derived from it.

##### Solving the system of equations

When considering all constraints, finding all solutions by hand becomes extremely tedious as soon as more than three or four variables and predicates are involved — MaxPool's structural domain alone (see the example above) already has around a dozen predicates over eight variables. One possible solution is to use a constraint solver such as z3. An example is given in this [Jupyter notebook](https://colab.research.google.com/drive/14wIuDS6uYxWioI7IzVirBsSdbtp0xbct?usp=sharing).

**A caveat on tractability.** The Add and Div examples above involve only linear arithmetic over few, largely independent variables, which SMT solvers handle essentially instantly — they are not representative of every SONNX operator. MaxPool's own output-size constraint (\*\*) multiplies and floor-divides several attributes together ($\text{dilations}\times(\text{kernel\_shape}-1)$, division by $\text{strides}$), which pushes the problem into **non-linear integer arithmetic**, a substantially harder decision fragment for Z3: solving can be slow in practice and is, over an unbounded domain, undecidable in general. Consequences follow directly from mechanisms already introduced in this document, rather than requiring a new one:
- the usage-domain bounds of the previous section (rank ≤ 4, size ≤ 1000, etc.) should be applied to the solver's search *before* enumeration is attempted, not only used afterwards to interpret its output — i.e., this is a bounded-model-checking use of the solver, reasoning over a bounded integer range rather than the unbounded domain implied by the operator's type;
- for each operator, the raw class count returned by the solver — before and after pruning — should be recorded in that operator's test-design record. This turns the informal expectation stated earlier ("the domain is pretty complex, so a strict application of the test generation strategy will end up with a very large number of test cases") into a measured fact that can be checked against the actual test budget, and flags early when an operator's constraint system needs to be simplified (e.g. by fixing some structural attributes) before the solver is asked to enumerate it. This measurement is an open action item for MaxPool and should be filled in as the solver notebook is run against its full structural constraint set, rather than assumed;
- for a floating-point equality class derived from an arithmetic constraint (e.g. an equality class of the form $x_1+x_2=c$, or one produced by the `nextdown`-based boundary rewriting above), the existence of an exact representable solution is **not** guaranteed the way it is for integers: rounding can mean no pair of floating-point values satisfies the equality exactly, or that the actual solution set differs from what real-arithmetic reasoning would suggest. The solver must therefore be invoked using a floating-point theory (e.g. Z3's `FP` sort) rather than real or rational arithmetic for any predicate over floating-point variables, and the notebook referenced above should state explicitly which theory it uses for each operator — silently solving in real arithmetic and rounding the result afterward is not equivalent to solving directly in floating-point arithmetic.

##### Eliminating irrelevant cases

We have already considered the case where some predicates are incompatible.  For instance, when the predicates defines the left and right bounds onf an interval,  they cannot be simultaneous true since a value cannot be simultaneously the minimum and the maximum of the domain (except for intervals reduced to a unique value).
Those combinations can be eliminated when building the set of constraints. However, if some solver is used (e.g., Z3), they can be left since they will easily be detected as "non satisfiable" (NONSAT) by the solver.

The second case concern combinations that are considered not pertinent with respect to the test objectives. By "not pertinent", we mean that *we do not expect a particular erroneous behaviour for that specific combination of values*.
As this elimination criterion is based on engineer judgment, for each excluded combination, the test designer must record the following data to justified their choice:

| Field                          | Content                                                                                                                                                                                                                                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Combination**                | The predicate conjunction being excluded (e.g. $p'_1 \wedge p''_2 \wedge p''_3 \wedge p'_4$)                                                                                                                                                                                                                     |
| **Fault model assumed absent** | The specific implementation pattern that *would* make this combination reveal a distinct error, stated concretely enough that a reviewer could disagree with it (e.g. "an implementation that special-cases `dilations[0]==dilations[1]` simultaneously with `pads[0]==0`" — not just "no expected interaction") |
| **Disposition**                | `pruned` (excluded from the test set), `covered-by` if the case is excluded because another retained test case is expected to reveal the same class of error (and cite that test's ID), or `pairwise-coverage` if the case is retained (or excluded) purely to satisfy the systematic technique of the same name below, without an individually stated fault model |

*The table below is an **illustrative worked example**, not a reviewed engineering record for MaxPool: the specific combinations and "fault model assumed absent" claims are constructed here to show how the field would be filled in, and have not been checked against MaxPool's actual implementation landscape. Each entry must be independently verified — or replaced with a genuine assessment — before this table is relied upon as the operator's real pruning justification.*

Applying this to the pads/dilations/kernel system in the **MaxPool** example, we obtain:

| Combination                                                                                                           | Fault model assumed absent                                                                                                                                                                                                  | Disposition       |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `dilations[0]=1 ∧ dilations[1]∈]1,max[ ∧ strides[0]=1 ∧ strides[1]=1`                                                 | No implementation is expected to handle asymmetric dilation across the two spatial axes differently when strides are both at their minimum — dilation handling is usually implemented per-axis independently                | pruned            |
| `pads[0]=pads[2] ∧ pads[1]≠pads[3]` (symmetric on one axis, asymmetric on the other)                                  | Covered by the explicit symmetry/asymmetry tests (SYM-01..04); no additional interaction between axes is expected since padding is applied per-axis                                                                         | covered-by SYM-02 |
| `strides[0]=strides[1] ∧ dilations[0]=dilations[1]` at their respective interior (non-boundary) values simultaneously | No known accumulator, index-computation, or output-size-rounding pattern couples strides and dilations of *different* axes — output size for each axis is computed independently per the operator's own output-size formula | pruned            |

**Combining classes across the Test completeness criteria.** § Test completeness lists six criteria (data type, shape, broadcasting, pre-conditions, functional specification, implementation-risk patterns), each currently built by a separate procedure over its own predicate set. This is not a different problem from the one just described, once stated precisely: where two criteria's classes are meant to interact — for instance, a special value (from the type/special-value system) landing on a structural boundary (from the pre-condition system), the MaxPool index case identified in § Cross-domain dependency check — the correct move is to express that interaction as additional predicates added to a single, *shared* $P$ for the variables concerned, exactly as the **Abs** example already does by folding both functional branches and special values into one predicate list ($p'_1$ through $p'_7$, § Discontinuities). Once unified, the boundary/interior decomposition (§ Equivalence classes based on the input domain) enumerates the joint classes automatically — nothing beyond that machinery is needed. The resulting, typically large, set of combinations is then reduced using exactly the pruning mechanism above — `pruned` or `covered-by`, each with a recorded fault-model-assumed-absent justification — rather than by a separately named reduction policy.

Two criteria whose classes are never unified into a shared $P$ simply remain independent test vectors, each holding the other criterion's inputs at an arbitrary representative value (§ Selecting a representative value within a class). This is the default, and it needs no separate name or policy beyond "these predicate sets were never joined" — it is not a weaker or fallback form of coverage, just the case where no interaction was identified worth expressing jointly.

**A caution when unifying an ordered domain predicate with a special-value predicate.** The boundary/interior rewriting of § Equivalence classes based on the input domain assumes an *ordered* domain: every value is classified by comparing it against a boundary via `≤`, `≥`, `<`, or `>`. Under IEEE 754, every relational comparison involving NaN evaluates to false, so a NaN value satisfies neither $p'_i$ nor $p''_i$ for any ordinary ordered predicate $p_i$ — it is silently excluded from the enumerated combinations unless captured by its own explicit equality predicate, exactly as the **Abs** example already does ($X[i]=\text{NaN}$, § Discontinuities). When unifying an ordered domain predicate with a special-value predicate over the same variable, check whether the special value falls outside every $p'/p''$ pair produced by the ordered predicates, and if so add it as an explicit sibling equality predicate rather than assuming the boundary/interior rewriting already captures it. `±inf` does not have this problem, since it participates normally in ordered comparisons and is correctly captured as one of the boundary values of an unbounded domain.

**Pairwise (or t-wise) coverage as a systematic pruning technique.** The pruning table above requires the test designer to state, one combination at a time, an implementation pattern whose absence is being assumed. For a large unified predicate set this becomes as tedious as the enumeration itself, and coverage then depends entirely on whoever is doing the pruning having an informed opinion about every combination. Where no such opinion exists yet, generating the retained set from a **covering array** — guaranteeing every *pair* of class choices across the unified predicates is exercised at least once (t=2; a higher t where the budget allows) — is a mechanical alternative to one-at-a-time manual pruning, not a rival composition tier sitting next to it. Combinations kept this way are recorded with the `pairwise-coverage` disposition introduced in the field table above, so a reviewer can see, per test, whether it was kept because of a stated fault model or because it filled a pairwise-coverage obligation.

##### Validating pruning decisions via mutation testing

§ Eliminating irrelevant cases requires the test designer to record, for each pruned combination, the implementation pattern whose *absence* is being assumed. That record is a documented belief, not a checked fact — nothing so far verifies that the retained test set would actually catch the pattern if it were present. **Mutation testing** closes this gap: a set of plausible implementation bugs is injected into the reference implementation, and the retained test set is run against each mutant. A mutant that survives (produces the same output as the un-mutated reference on every retained test) is evidence that either the pruning was too aggressive, or that an existing `covered-by` disposition does not actually cover what it claims to.

*The examples of candidate mutants below (off-by-one on a boundary index, swapped `strides`/`dilations`, a wrong rounding direction, a missing `-inf` special case) are illustrative of the kind of bug this technique targets, not a reviewed or exhaustive list of actual MaxPool implementation risks — a real mutant set should be derived from the operator's actual reference implementation and known defect history.*

This is a natural complement to, not a replacement for, the equivalence-class approach: the classes decide *which* inputs are worth trying; mutation testing checks whether the ones actually kept are sufficient. Each `pruned` or `covered-by` entry in the table of § Eliminating irrelevant cases should ideally have at least one corresponding mutant designed specifically to test whether the assumed-absent fault model, if it were present, would in fact go undetected.

#####  Cross-domain dependency check

Before applying equivalence-class decomposition, the test designer inspects the informal spec and answers the following for each operator:

1. **Accumulation.** Does the operator compute any running sum, product, or other combining operation over more than one input element (reduction, convolution, matmul, pooling-with-average, etc.)? If yes: does the accumulator's intermediate or final range depend on *both* the number of elements combined (a structural/size variable) *and* their values? If yes, the size and value domains for that accumulation **cannot** be tested independently — a combined boundary case (max size × max value, or the size threshold at which overflow becomes possible for boundary values) must be added explicitly.
2. **Index computation.** Does the operator compute an output index or position from input structural parameters (as MaxPool does)? If yes: can a value-domain choice (e.g. `-inf`, ties, `NaN`) interact with a structural boundary (e.g. a window at the tensor edge, or a null dimension) to produce an *additional* distinct failure mode beyond testing each independently? If yes, add the combined case explicitly (this is already anticipated informally by IDX-03 in the MaxPool test set, which puts a special-value-adjacent window at a padding boundary — this check makes that inclusion a required step rather than an incidental one).
3. **Parameter coupling in output-size formulas.** Where an operator's output size or shape is computed from more than one attribute (e.g. MaxPool's `kernel_shape`, `strides`, `pads`, `dilations` all appearing in the same formula), does the formula multiply, divide, or otherwise combine two attributes rather than applying them independently per axis? If yes, boundary values of those attributes must be tested in combination, not only individually at their own boundaries.
4. **Broadcasting.** For operators that support broadcasting, can a broadcast-from-1 axis (§ Broadcasting-specific domains) combined with a special value on the size-1 operand produce a distinct failure mode? If yes, add the combined case explicitly.

The result is a short **dependency table** per operator:

| Candidate dependency                   | Found? | If yes: combined boundary test added |
| -------------------------------------- | ------ | ------------------------------------ |
| Accumulator range vs. tensor size      | —      | —                                    |
| Output index vs. special/tied values   | —      | —                                    |
| Output-size formula attribute coupling | —      | —                                    |
| Broadcasting vs. special values        | —      | —                                    |

*The table below, applied to MaxPool, is again an **illustrative worked example** — the "found"/"not found" assessments are plausible but have not been checked against a real implementation, and must be independently verified before being treated as the operator's actual dependency record.*

Applied to the **MaxPool** operator, it gives:

| Candidate dependency | Found? | If yes: combined boundary test added |
|---|---|---|
| Accumulator range vs. tensor size | No — MaxPool has no accumulator; it is a pointwise-selection operator, not a combining one. Overflow-style dependencies do not apply. | — |
| Output index vs. special/tied values | **Yes** — a `-inf`/`NaN`/tied-max value located in a window that touches a padding boundary could interact with index computation in ways that testing "special values" and "index boundaries" separately would miss | New test: max-tie or `NaN` element positioned in the window nearest the padded edge, distinct from both SPEC-0x (values only, interior window) and IDX-03 (boundary index, ordinary values only) |
| Output-size formula coupling (`kernel_shape` × `strides` × `pads` × `dilations`) | **Yes** — constraint (\*\*) already combines all four attributes multiplicatively/additively in one formula per the existing spec text | Already partially covered by ATTR-06/07/08/09; this check confirms those are necessary rather than optional and flags that a *dilations > 1* variant of ATTR-08/09 (output-size boundary) has not yet been generated and should be added |
| Broadcasting vs. special values | N/A — MaxPool does not support broadcasting between its inputs | — |

##### Maintaining a cross-operator pattern catalog

The Cross-domain dependency check above and the Special values / Discontinuities tables (§ Operator-specific constraints) are each derived from scratch for every operator analysed so far. In practice, an interaction pattern discovered on one operator is very likely to recur on others that share structure — the MaxPool finding that a special value positioned at a structural (padding) boundary can produce a failure mode invisible to either check in isolation is just as plausible for AveragePool, Conv, or any other sliding-window operator.

To avoid re-deriving the same insight independently each time, the dependency-check questions, the special-value/discontinuity tables, and any interaction discovered by mutation testing (§ Validating pruning decisions via mutation testing) or by an actual specification-vs-implementation discrepancy (§ Test objectives and scope in SONNX) should be maintained as a single, shared, living catalog across all SONNX operators rather than as per-operator, one-off tables. When a new pattern is found on operator $A$, the catalog is updated, and every other operator sharing the relevant structural feature (e.g., "sliding window with padding," "reduction with an accumulator") is flagged for re-review against the new entry, rather than waiting for the same pattern to be independently rediscovered.

##### Symmetry and asymmetry

Consider the the `pads` parameter for the convolution. Besides testing the boundaries of the domain  (in 2D), we may also exercice certain relations between the parameters. For instance, we may want to exercize symmetric and asymmetric padding. It may be the case that by generating tests with respect to the bounds of the domain will also cover symmetric and asymmetric configurations (simply because we will generate a test for all combinations of min and max values for all paddings), but this is fortuitous. A good test strategy shall make this test against symmetry explicit.

##### Test completeness

In our context, a test set is deemed complete relative to an operator specification if :
1. It covers every equivalence classes built upon data-types, for all data types supported by the operator
2. It covers every equivalence classes built upon the tensor shapes (rank and sizes)
3. It covers every equivalence classes built upon broadcasting
4. It covers every equivalence classes built upon the operator pre-conditions (so-called "constraints" in the specification)
5. It covers every equivalence classes built upon the operator functional specification
6. It covers every important implementation-risk pattern

How classes built for different criteria above combine into concrete test vectors — and how the resulting combinations are kept to a tractable set — is addressed in § Eliminating irrelevant cases (unify into a shared predicate set where an interaction matters, then prune). Criterion 3 (broadcasting) is built as described in § Broadcasting-specific domains. Criterion 6 (implementation-risk patterns) is not built by a dedicated construction procedure of its own; it is populated by two other mechanisms described later in this document: § Maintaining a cross-operator pattern catalog supplies patterns known in advance from other operators, and § Validating pruning decisions via mutation testing surfaces patterns the retained test set fails to catch, which are then fed back into that catalog.

##### Risk-based prioritization of test generation

§ Test completeness lists six criteria as jointly necessary for a complete test set, with no distinction of priority between them. In practice, test-development time is finite, and the strategy should say in what order classes get generated when the budget does not allow generating all of them at once. A reasonable default ordering, from highest to lowest expected return:
1. classes flagged by the Cross-domain dependency check or the Symmetry and asymmetry check (named, specific interaction risk);
2. boundary and vertex classes (edges and corners of the input domain, § Equivalence classes based on the input domain) and special-value / discontinuity classes (§ Operator-specific constraints);
3. pairwise-coverage combinations (§ Eliminating irrelevant cases) not already covered by (1) or (2);
4. "rest of the domain" interior classes with no known special role.

This ordering does not change what "complete" means (§ Test completeness still defines that), only the order in which an incomplete-but-in-progress test set is built, so that a test campaign interrupted partway through has spent its budget on the higher-risk classes first.

### Domains
#### Type-specific domains
Type-specific domains are defined with respect to the data type.  They are independent of the semantics of operators.
##### Floating point numbers
For instance, for floating point number, the domains are the following (some domains are singletons):
- NaN
- +inf, -inf
- +0, -0
- subnormal values (also called "denormalized" values)
	- for float:
		- min positive: $2^{-149} \approx 1.45\times 10^{-45}$
		- max positive: $(1−2^{−23})\times 2^{−126} \approx 1.18×10^{-38}$
	- for double:
		- min positive: $2^{-1074} \approx 4.9×10^{-324}$
		- max positive: $(1−2^{−52})\times2^{−1022} \approx 2.23\times 10^{-308}$
- normal value
	- for float:
		- min positive: $2^{-126} \approx 1.18×10^{-38}$
		- max positive: $(1-2^{-24})\times 2^{128} \approx 3.40 \times 10^{38}$
	- for double
		- min positive: $2^{-1022} \approx 2.23\times 10^{-308}$
		- max positive:  $(1-2^{-53})\times 2^{1024} \approx 1.80 \times 10^{308}$

Note: Several reasons make subnormal values worth considering:
- This is "where" underflows may occur (values rounded to 0)
- The relative precision of subnormal values is lower than for normal number. Indeed, normal numbers have *fixed relative* precision whereas subnormal numbers have *fixed absolute* spacing near zero. Therefore, their relative precision becomes worse as they get closer to zero. (However, note that accuracy is not addressed in the specification)
- The treatment of subnormal values may depend on the compiler / hardware platform
- The subnormal hey may expose corner cases in comparisons and branching (e.g., comparison to 0)

As stated in § Selecting a representative value within a class, these type-specific classes are not an independent axis to be tested in isolation from the domain-boundary classes of § Equivalence classes based on the input domain: for a floating-point variable, they are precisely the source from which the *interior* representative of a boundary/interior predicate class is drawn, once the two predicate sets are unified as described in § Eliminating irrelevant cases.

The domain is pretty complex, so a strict application of the test generation strategy will end-up with a very large number of test cases, even for an operator with two arguments. A strategy must be defined to restrict the number of combinations to be considered.

##### Integer numbers
For integer numbers (uint, int):
- min int  (``minInt`` for the considered integer type)
- max int (``maxInt`` for the considered integer type)
#### Structure-specific domains

In the context of SONNX, a structure is a tensor characterized by its shape (number of dimensions -- or rank -- and a size per dimension).  The domains are defined with respect to these two parameters.

Tests shall consider
- the rank
	- scalar tensors (rank = 0)
	- vector (rank=1)
	- matrix (rank=2)
	- all other ranks
- the dimension
	- null-tensors (at least one dimension with size zero)
	* multi-dimensional tensors reduced to a lower dimension tensor, such as a 2-dimensional tensor of shapes 1x1 (a scalar), 1xn (a line vector), nx1 (a column vector)
	* all other dimensions

Note that "all other ranks" and "all other dimensions" are themselves upper-bounded in practice by the usage-domain predicates introduced in § The case of unbounded variables (e.g. rank ≤ 4); they are not open-ended classes, and the same predicate-splitting and justification-recording rules apply to them.

#### Broadcasting-specific domains

*This section is a first proposal illustrating the kind of construction broadcasting needs; unlike the type- and structure-specific domains above, it has not been cross-checked against SONNX's actual broadcasting specification and should be reviewed before use.*

Broadcasting is listed as a mandatory completeness criterion (§ Test completeness, item 3), but, unlike data types and tensor structure, it had no construction procedure of its own until this revision. Broadcasting classes should be built from the alignment rules applied per axis when combining tensors of different shapes:

- **Per-axis dimension pairing.** For each pair of aligned axes (from the trailing dimension inward), the two input sizes are either:
	- equal (no broadcasting needed on that axis) — the "no broadcast" class,
	- one of the two is exactly 1 (that axis is stretched to match the other) — the "broadcast-from-1" class, with sub-classes for which of the two operands is the size-1 side,
	- neither equal nor 1 — incompatible, and the operator must reject the input (an error class, not a computed-result class).
- **Rank mismatch.** When the two operands have different ranks, the shorter shape is implicitly left-padded with size-1 axes before per-axis pairing. This produces a further class distinguishing "same rank, per-axis broadcasting only" from "different rank, requiring implicit alignment" — the two exercise different code paths in most implementations (an explicit loop over declared axes vs. an implicit padding step) and are not equivalent from a testing standpoint even where the per-axis outcome is otherwise identical.
- **Degenerate cases.** A rank-0 (scalar) operand broadcasting against a rank-$n$ tensor, and a null-dimension (size-0) axis paired against a size-1 or size-$n$ axis, are boundary classes of the above in the same sense as § Equivalence classes based on the input domain's boundary/interior split, and should be enumerated explicitly rather than left to be incidentally covered by the general tensor-shape classes of § Structure-specific domains.

As with every other domain in this document, broadcasting classes are unified with the operator's other predicate sets (§ Eliminating irrelevant cases) where an interaction is expected — for instance, a broadcast-from-1 axis combined with a special value on the size-1 operand, which § Cross-domain dependency check now asks about explicitly for operators that support broadcasting.

#### Operator-specific constraints

Operator-specific constraints are derived from the operator informal specification.

We consider two general cases:
- the case of "special values", which consist in identifying of values playing a "special" role in the specification of the operator
- the case of "properties", which consists in identifying of special properties of the operator.
##### Special values

The test designer has to consider the existence of:
- neutral values
- absorbing values
- dominating values

Such special values depend on the operator semantic. They may be identified by a systematic analysis of the specification, but more certainly by using the knowledge of the semantics of the operator.

For instance:
- for **Add**
	- 0 :  neutral element
- for **Mul** , **MatMul**
	- 1, or $I$, the identity tensor  :  neutral element
	- 0, or the null tensor  :  absorbing element
- for **Div**
	- 1 at denominator : neutral element
- for **Max** , **Maxpool**
	- +inf : absorbing value
	- -inf : neutral element
- etc.

Note: It is not clear if we should consider these values as specific if they do not determine a specific "branch" or singularity in the specification. For instance, the $Add$ operator does not show any behavioural singularity for value 0.
##### Discontinuities

Values representing a "discontinuity" in the function behaviour (in the general and mathematical sense), including exceptional cases and error cases.

- for **Div**
	- 0 at denominator :  discontinuity (division-by-zero)
- for **Relu**
	- 0 :  Relu threshold
- for  **Abs**
	- The operator is specified using a series of "if" as follows ($i$ is any multi-index compatible with the structure of tensor $X$):
				$$
				Y[i] =
				\begin{cases}
				\text{NaN} & \text{if } X[i] = \text{NaN} \\
				\text{+Inf} & \text{if } X[i] = \pm \text{Inf} \\
				\text{+0} & \text{if } X[i] = \pm \text{0} \\
				-X[i] & \text{if } X[i] \lt 0  \\
				X[i] & \text{otherwise}
				\end{cases}
				$$
	- This leads to the following constraints
		- $p'1 : (X[i] = NaN)$
		- $p'2 : (X[i] = +\text{Inf})$
		- $p'3 : (X[i] = -\text{Inf})$
		- $p'4 : (X[i] = -\text{0})$
		- $p'5 : (X[i] = +\text{0})$
		- $p'6 : (X[i] < 0) \wedge (X[i] \neq \text{-Inf})$
		- $p'7 : (X[i] > 0) \wedge (X[i] \neq \text{+Inf})$
## Additional tests

The following tests do not apply a strategy based on equivalence classes. Instead, they apply a strategy based on the verification that some property holds between the results of multiple execution.

For instance, the following properties may be verified
- commutativity (**Add**, **Mul**, **Max**, etc.), e.g,
	- **Add**(a,b)=**Add**(b,a)
- linearity, e.g.,
	- **Conv**(X+Y,W) = **Conv**(X,W)+**Conv**(Y,W)
- invariance per rotation
	- **Sin**($x$)=**Sin**($x+2k\Pi$),
- invariance per translation
	-  **maxpool**: invariant** of the max index when all elements of the tensor are added a constant $c$ (assuming no over/under flow)
- invariance per scaling.
	-  **maxpool**: invariant** of the max index when all elements of the tensor are multiplied by a  stricly positive constant (assuming no over/under flow)


### On values and indexes...

When designing the test, care shall be taken to verify that the result of the computation of some input value $X[i]$ for multi-index $i$  is actually stored at the appropriate output $Y[i']$. more generally, we are checking that $Y[i'_1]=f_1(X[i_1], X[i_2], ..., X[i_n]$, $Y[i'_2]=f_2(X[i_1], X[i_2], ..., X[i_n]$, etc.) which means that we are testing that the values are correct in the correct for the correct multi-indexes.

As the indexes $i'$ are not explicit outputs of the operator, the test must be designed so that the value of $i'$ can be derived from the observation of the output tensor $Y$. For instance, when testing the **Mul($X1$,$X2$)** operator, the contents of $X1$ and $X2$ must be such that all expected values of $Y$ are different. Using $X2[i]=0$  for all $i$ will, for instance, not satisfy this property since all $Y[i']$ will be equal to zero so there will be no way to check that the 0 at $Y[i']$ have been computed from the appopriate $X1[i]$ and $X2[i]$ (in that case, $i$ = $i'$).

The same constraint applies when dealing with structural operators such as **Flatten**. Note that the explicit property stating that the **Flatten** operator must preserve the values of the input tensor (invariance) is not sufficient since the output may well preserve the input values but place them at inappropriate indexes.

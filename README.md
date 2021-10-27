# Safe-Linear and Safe-Monotonic
---
This is the test code for experimental results in the paper *Best Arm Identification (BAI) with Safety Constraints*([arxiv link here]).

Here we outline 2 major algorithms to identify the best arm within the multi-arm bandit setting without violating a specified safety constraint.

**Safe-Linear** is tailored for linear reward and safety models, while **Safe-Monotonic** is applied on monotonic reward and safety models.

## Usage
---
For linear case, you should specify a bandit instance by using the `MAB` class from `CBAI`, then run the Safe-Linear on this instance generated. Otherwise, you may generate any random instance using the `instance_generator` from `utility.py`
You may call any of the functions above via:
`from CBAI import MAB, Algo, instance_generator`

Next, you may try out monotonic models by either using our sigmoid function or your own function (make sure its **monotonicity** is well behaved). Construct a particular instance following the format outlined in our notebook. Then you may call
`from CBAI import safe_monotonic_bai`.

Feel free to play around with the code if you need it for your research purpose. A simple and fun way to learn about the code is to run the ipython notebook here.
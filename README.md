# dynamic-program-embedding
This is the code for our ICLR'18 paper:
- Ke Wang, Rishabh Singh and Zhendong Su, [Dynamic Neural Program Embedding for Program Repair](https://arxiv.org/abs/1711.07163), 6th International Conference on Learning Representations, 2018. 

Please cite the above paper if you use our code. </br></br>
The code is released under the [MIT license](https://github.com/yujiali/ggnn/blob/master/LICENSE).

<h3>Notes</h3>

+ Training.py constains Variable Trace Model
+ StateTraining.py constains State Trace Model
+ HybridTraining.py constains Dependency Model
+ For all three models, one would have to prepare the input traces for training program. Please refer to the source code for each model on input format. For dependency model a couple of mask tensors are also needed.

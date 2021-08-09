# Elastic Weight Consolidation (EWC) Implementation - Tensorflow 2
During or after learning new skills, humans are able to retain most of the skills they had learned prior. This is a difficult thing for computers to do, ANN forget tasks they have previously learned during and after the learning of new ones. In order to mitigate this effect, Kirkpatrick et al in their popular work https://arxiv.org/abs/1612.00796 introduced the Elastic Weight Consolidation (EWC) technique. It measures the level of importance of each model parameter to the prior task in the form of Fisher Information Matrix. This then forms the bases of an imposed loss penalty on the training of the new task. 

The available repositories have implemented this technique in tf1, keras and pytorch. Here, I implemented it using tensorflow 2 (tf2) with custom training flow using tensorflow GradientTape and enger execution.

Below are plots from the sample usage code in the `demo` notebook.

<br>

<img src=https://github.com/stijani/elastic-weight-consolidation-tf2/blob/main/images/bars.png width=400 height=500 />

<br>

<img src=https://github.com/stijani/elastic-weight-consolidation-tf2/blob/main/images/lines.png width=400 height=500 />

<br>

Please reach out if you do find any issues in the implementation logic, code or have an advice to make things better.

本仓库是基于huggingface的示例`diffusers/examples/textual_inversion`改编而成的Textual Inversion方法，你可以执行`MTI_train`脚本进行训练。主要改动如下：

* 可自定义提示词，你可以将提示词存储到`.json`文件，并放入`anno`文件夹，具体参考`anno`文件夹的示例。如果不指定提示词，则默认使用原本的模板构建提示词。
* 可自定义嵌入的数量，修改嵌入数量的方法可参考`MTI_train.sh`文件中的`embeddings_number`设定。

This repository is an adaptation of Textual Inversion based on the example of huggingface ` diffusers/examples/textual_inversion `. You can execute `MTI_train` for training. The main changes are as follows:

* Customize the prompt words. You can store prompts in a `.json ` file and put them in the `anno` folder. For details, see the example in `anno` folder. If prompts are not specified, the original template will be used to build the prompts by default.
* Customize embeddings number. The method of modifying the number of embeddings can refer to the `embeddings_number` setting in the `MTI_train.sh` file.


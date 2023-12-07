# Micro-Tiny GPT Model FROM SCRATCH
This is our micro-tiny GPT model (😁 we are still learning), built from scratch and inspired by the innovative approaches of Hugging Face Transformers and OpenAI architectures. Developed during our internship at [3D Smart Factory](https://3dsmartfactory.csit.ma/), this model showcases our committed efforts to create an advanced AI solution.


<div align="center">
  <h5>Micro-Tiny-GPT</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/264447768-39a2a1a9-ca80-44cd-9d83-fd1358cb379b.png" width="200" height="200" />
</div>

## NOTE

**Important:** This project involves fine-tuning various GPT family models (small, medium, large, etc.) to develop two distinct chatbots: one for question and answer interactions [here](https://github.com/benisalla/fine-tune-GPT-chatbot) and another for context-based question and answer interactions [here](https://github.com/benisalla/fine-tune-gpt2_on_context_Q-A). Additionally, we have implemented a question and answer system based on the core transformer architecture which was initially used for translation tasks [here](https://github.com/benisalla/chatbot-based-translation-transformer).


**Note:** This is a learning project and does not represent a production-grade solution. The focus is on educational growth and experimentation.



<aside class="notice" style="background-color:#FFFFE0; border:2px solid #E0E000; padding:10px;">
**Note:**
This is an important notice. It provides additional information or a reminder.
</aside>





## Table of Contents

- [Introduction](#introduction)
- [OverView](#overview)
- [Data](#data)
- [Background on GPT](#background-on-GPT)
- [Model Architecture](#model-architecture)
- [Features of this project](#features-of-this-project)
- [Benefits of Utilizing MT-GPT](#benefits-of-utilizing-mt-gpt)
- [Explanatory Videos](#explanatory)
- [Training](#training)
- [Inference](#inference)
- [Fine Tuning](#fine)
- [Contributors](#contributors)
- [Contact Us](#contact-us)
- [About Me](#about-me)






# Introduction
Welcome to the Micro-Tiny GPT Model repository! This project is an exploration into building a compact GPT model from scratch, taking inspiration from the Hugging Face Transformers and OpenAI architectures. Developed during an internship at 3D Smart Factory, this model represents our dedication to creating advanced AI solutions despite our learning phase.






# OverView

the Following Readme file offers a thorough exploration of the meticulous process involved in building the foundational GPT model from its inception. Our journey covers vital phases, including data collection, model architecture design, training protocols, and practical applications. Throughout this chapter, we illuminate the intricate nature of developing such a potent language model and its profound potential across various applications in the field of natural language processing.









## Data

### WebText (OpenWebTextCorpus)

To train GPT, OpenAI needed a substantial corpus of 40 GB of high-quality text. While Common Crawl provided the necessary scale for modern language models, its quality was often inconsistent. Manually curating data from Common Crawl was an option but a costly one. Fortunately, Reddit provided a decentralized curation approach by design, proving to be a crucial innovation for creating the WebText dataset.

The WebText generation process can be summarized as follows:

1. Retrieve URLs of all Reddit submissions until December 2017 with a score of 3 or higher.
2. Deduplicate retrieved content based on URLs.
3. Exclude Wikipedia content, as OpenAI already had a separate Wikipedia dataset.
4. Further deduplicate the remaining content using an undisclosed "heuristic" cleaning method, including the removal of non-English web pages.

Neither the resulting corpus nor the source code for its generation was made public, which later inspired Aaron Gokaslan and Vanya Cohen to create the OpenWebText corpus.

### OpenWebText

OpenWebText2 is an enhanced version of the original OpenWebText corpus, covering all Reddit submissions from 2005 to April 2020, with additional months becoming available after the publication of corresponding PushShift backup files.


<div align="center">
  <a href="https://skylion007.github.io/OpenWebTextCorpus/">
    <h5>OpenWebText</h5>
  </a>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406287-b908e1a2-8650-48fd-a54e-d7ed79aed037.png" width="300" height="150" />
</div>


Due to resource constraints, it is important to note that we trained GPT on only a quarter of the OpenWebText dataset. This limitation in training data was necessary to optimize computational resources while still achieving significant language model performance.





### Data Preprocessing

We combined approximately 5 million files from the OpenWebText dataset, roughly equivalent to a quarter of the entire OpenWebText corpus. Subsequently, we performed the following steps:

1. We used the GPT tokenizer, also known as "r50k_base" to tokenize the dataset.
```
import tiktoken
tokenizer = tiktoken.get_encoding("r50k_base")

```

2. Following established conventions for dataset splitting, we divided the data into training and validation sets, allocating 80% for training and 10% for validation.
3. To optimize data management and efficiency, we stored the data as a binary stream in the 'train.bin' and 'val.bin' files.
```txt files to bin files
data_dir = "your data path"

for prefix in [list of folers in case you are using openwebtext]:
    f_dir = "dest dir"
    for idx, filename in enumerate(os.listdir(f_dir)):
        src_file = os.path.join(f_dir, f'f{idx+1}.txt')
        with open(src_file, 'r', encoding='utf-8') as f:
            content = f.read()
        dset = tokenizing(content)
        content = None
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        dest_file = os.path.join(f"{data_dir}/file_{prefix}/", f'f{idx+1}.bin')
        dtype = np.uint16 

        arr = np.memmap(dest_file, dtype=dtype, mode='w+', shape=(arr_len,))    
        arr[0:arr_len] = np.array(dset['ids'])
        arr.flush()
        print(f"✅ f[{idx+1}].txt saved successfully to f{idx+1}.bin")
```

```Combine all bins to on large bin (that is silly but i don't have resources)

for split_name in ["test", "val", "train"]:
    data_dir = "your data dir"
    f_dir = "dir to where you want to put your data"
    with open(os.path.join(f_dir, f'{split_name}.bin'), 'wb') as outf:
        for idx, filename in enumerate(os.listdir(data_dir)):
            src_path = os.path.join(f"{data_dir}", filename)
            with open(src_path, 'rb') as input_file:
                file_contents = input_file.read()
                outf.write(file_contents)
    print(f"Concatenation of {split_name} complete.")

```





Our dataset is currently accessible on Kaggle:

<div align="center">
  <a href="https://www.kaggle.com/datasets/benallaismail/gpt-data">
    <h5>Quarter of OpenWebText by Ben Alla Ismail</h5>
  </a>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406286-22c3d2ab-df41-4c49-981f-ae153445ce5b.jpeg" width="600" height="300" />
</div>











## Background on GPT

The Generative Pre-trained Transformer 2 (GPT), developed by OpenAI, is the second installment in their fundamental series of GPT models. GPT was pretrained on the BookCorpus dataset, consisting of over 7,000 unpublished fiction books of various genres, and then fine-tuned on a dataset comprising 8 million web pages. It was partially unveiled in February 2019, followed by the full release of the 1.5 billion parameter model on November 5, 2019.


<div align="center">
  <h5>GPT Model</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406289-a66928b2-b651-4fb5-81b2-f9429685d9d0.jpg" width="600" height="300" />
</div>


GPT represents a "direct scale-up" from its predecessor, GPT-1, with a tenfold increase in both the number of parameters and the size of the training dataset. This versatile model owes its ability to perform various tasks to its intrinsic capacity to accurately predict the next element in a sequence. This predictive capability enables GPT to accomplish tasks such as text translation, answering questions based on textual content, summarizing text passages, and generating text that can sometimes closely resemble human style. However, it may exhibit repetitive or nonsensical behavior when generating long passages.

There is a family of GPT models; below, we can see the pretrained GPT model family:


<div align="center">
  <h5>GPT Model Family</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406288-e0dc48f8-a5e1-4807-8184-39ca3e2a88c5.jpg" width="700" height="300" />
</div>















## Model Architecture

The architecture of GPT, a groundbreaking language model, represents a notable evolution of deep learning-based transformer models. Initially, it followed the traditional transformer architecture with both encoder and decoder components, but subsequent research simplified the design by removing one.

<div align="center">
  <h5>GPT Relationship with Transformers</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406295-de6880cf-7d77-4b16-bcb5-d58b950e3404.jpg" width="700" height="300" />
</div>



This led to models with exceptionally high stacks of transformer blocks and massive volumes of training data, often requiring significant computational resources and costs. This chapter explores the architecture of GPT and its relationship with transformers, highlighting innovative developments that shaped its evolution into a powerful language model.


<div align="center">
  <h5>Full GPT Architecture</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406294-83845cfb-f442-478f-9993-3412225dabb7.png" width="350" height="600" />
</div>













## Features of this project

- **Micro-Tiny Scale:** This GPT model is intentionally designed to be on a micro-tiny scale, showcasing our learning journey and commitment to innovation.

- **Inspired by Industry Leaders:** Influenced by the Hugging Face Transformers and OpenAI architectures, we've incorporated industry best practices into our model's design.

- **Internship Project:** Developed during our internship at 3D Smart Factory, this project reflects real-world experience and hands-on learning.











## Benefits of Utilizing MT-GPT

Within the realm of Micro-Tiny GPT notebook, there exists a multitude of advantageous applications:

1. Enhanced Textual Performance: Leveraging the ability to train MT-GPT on larger datasets can significantly improve its text generation capabilities.

2. ABC Model Integration: You will find a demonstrative example of integrating MT-GPT with the ABC model in my GitHub repository, showcasing its adaptability to diverse frameworks.

3. Voice Training Capabilities: MT-GPT can be trained on voice data, opening up opportunities for voice-related applications.

4. Sequential Problem Solving: The versatility of MT-GPT extends to addressing various sequential challenges within the field of AI.



## Explanatory Videos

I've created explanatory videos to help you understand the intricate link between the architecture and the code, as it might not be immediately obvious.

<div align="center">
  <a href="https://drive.google.com/file/d/14V5PbIBW3fgVWAvB1yux_lGHL6ULtdcD/view?usp=drive_link">
    <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288115000-0db1ec7c-06a9-4801-8ae0-900b2c0382e2.png" width="700" height="300" alt="Explanatory Video"/>
  </a>
</div>






## Training

<div align="center">
  <h4>Loss function</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406292-81226d87-8166-4f6d-b913-baa419f794ff.png" width="600" height="300"/>
</div>







## Inference

<div align="center">
  <h4>Example 1</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406291-a362a3cf-3886-40d8-a1cc-71f448616a22.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Example 2</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288919714-b1151a7c-f15e-46d1-b39b-7035cb864406.png" width="700" height="300"/>
</div>
<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288919814-f2983df6-a849-49ef-9bba-e47b88266de2.png" width="700" height="300"/>
</div>
<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288919922-62558357-7578-45e1-b762-18a9327f6890.png" width="700" height="300"/>
</div>






## fine tuning MT-GPT

  For detailed information, refer to the project [Fine-Tuning Pretrained GPT-2](https://github.com/benisalla/fine-tune-gpt2_on_context_Q-A), where GPT-2 has been fine-tuned specifically for context-question-answer scenarios.

<div align="center">
  <h4>Example 1</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288918563-bd3fc6ca-d24d-4358-b5f8-22ce6afc09b8.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Example 2</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288918548-b812b6e0-d3bf-4aea-8c89-370a4e79f7e2.jpeg" width="700" height="300"/>
</div>
<div align="center">
  <h4>Example 3</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288918495-a1bca43c-82f1-4322-ae98-18f76836aaab.jpeg" width="700" height="300"/>
</div>




## Contributors

Acknowledge the significant contributions of your colleagues, attributing specific roles and responsibilities to each individual:

- [Ben Alla Ismail (me 😃)](https://github.com/benisalla)
- [Agrat Mohammed](https://github.com/agrat)
- [Mhaoui Siham](https://github.com/siham)
- [Souhayle ou-aabi](https://github.com/Souhayle-ou-aabi)
- [Daoudi Ayoub](https://github.com/Daoudi-Ayoub)





## Contact Us
For inquiries or suggestions, please contact:
- Project Lead: Ben alla ismail ([ismailbenalla52@gmail.com](mailto:ismailbenalla52@gmail.com))
- Co-lead: mhaoui Siham ([mahouisiham@gmail.com](mailto:mahouisiham@gmail.com))




## About Me

🎓 I'm Ismail Ben Alla, and I have a deep passion for neural networks 😍. My mission is to assist neural networks in unraveling the mysteries of our universe.</br>
⛵ I'm an enthusiast when it comes to AI, Deep Learning, and Machine Learning algorithms.</br>
✅ I'm an optimist and a dedicated hard worker, constantly striving to push the boundaries of what's possible.</br>
🌱 I'm committed to continuously learning and staying updated with advanced computer science technologies.</br>
😄 I absolutely love what I do, and I'm excited about the endless possibilities in the world of AI and machine learning!</br>

Let's connect and explore the fascinating world of artificial intelligence together! 🤖🌟


<div align="center">
  <a href="https://twitter.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
  <a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="ismail-ben-alla-7144b5221/" height="30" width="40" />
  </a>
  <a href="https://instagram.com/ismail_ben_alla" target="blank">
    <img src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="ismail_ben_alla" height="30" width="40" />
  </a>
</div>






<div align="center">
  <h4>You are about to witness some pure magic ✨🎩 !! Ta-da!</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/270190847-0b3ee23b-c082-483e-9e12-8b15a1b8f0a3.gif" width="500" height="300"/>
</div>

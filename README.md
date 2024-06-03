# Ultrasound-Report-Generation
This is the code for "Ultrasound Report Generation with Cross-Modality Feature Alignment via Unsupervised Guidance".
We propose a novel framework for automatic ultrasound report generation, leveraging a combination of unsupervised and supervised learning methods to aid the report generation process. Our framework incorporates unsupervised learning methods to extract potential knowledge from ultrasound text reports, serving as the prior information to guide the model in aligning visual and textual features, thereby addressing the challenge of feature discrepancy. Additionally, we design a global semantic comparison mechanism to enhance the performance of generating more comprehensive and accurate medical reports.
![image](https://github.com/LijunRio/Ultrasound-Report-Generation/assets/91274335/63fe3ae3-293a-45b1-af9a-099468c644fc)

## Main Result
![image](https://github.com/LijunRio/Ultrasound-Report-Generation/assets/91274335/c216ef5e-8bea-4ca5-8214-de339a136861)

## Implementation
### Setting
- set the hyperparameter and path in ./KMVE_RG/config.py.

### Run clustering
- Run ./knowledge_Distiller/knowledge_distiller.py to obtain cluster labels.

### Run training process
- Run ./KMVE_RG/my_train.py to train the SGF.

## Data
The ultrasound dataset is available at https://drive.google.com/file/d/11Aw3_ETNBtfT1W7eWifbsaexFqSsM5BB/view?usp=drive_link.
To evaluate the performance of the proposed framework on different types of ultrasound datasets, we collected three different datasets of the breast, thyroid and liver. Specifically, the breast dataset consists of 3521 patients, the thyroid dataset consists of 2474 patients, and the liver dataset consists of 1395 patients.

![image](https://github.com/LijunRio/Ultrasound-Report-Generation/assets/91274335/d3bb3c79-7ad9-4cfa-92be-07a63734b4da)

## Citation
@inproceedings{li2022self,
  title={A self-guided framework for radiology report generation},
  author={Li, Jun and Li, Shibo and Hu, Ying and Tao, Huiren},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={588--598},
  year={2022},
  organization={Springer}
}

## Acknowledgement
This work was supported in part by Key-Area Research and Development Program of Guangdong Province (No.2020B0909020002), National Natural Science Foundation of China (Grant No. 62003330), Shenzhen Fundamental Research Funds (Grant No. JCYJ20200109114233670, JCYJ20190807170407391), and Guangdong Provincial Key Laboratory of Computer Vision and Virtual Reality Technology, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China. This work was also supported by Guangdong-Hong Kong-Macao Joint Laboratory of Human-Machine Intelligence-Synergy Systems, Shenzhen Institute of Advanced Technology.

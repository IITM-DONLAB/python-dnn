Data File Formats
==================

**Python-dnn** supports 3 file formats* for data:

1. ***NP*** :Numpy Format.
2. ***T1*** :Text File With One level header structure.
3. ***T2*** :Text File With Two level header structure

-----------------------------------------------------------------

###Numpy Format###

The dataset is stored as single file in binary format
Data file has:

```
	<json-header>
	<structured numpy.array>
	<structured numpy.array>
	..
	..
..
```

Fist Line is a **json-header** which Contains two paramters:
> * featdim : Dimention of input vector after flattening.
> * input_shape : Actual shape of input before flattening.
>
> eg: ``{"featdim": 784, "input_shape": [28, 28, 1]}``

Thereafter, each row contain a numpy structured Arrays (or Record Arrays) with vector as `'d'` and label as  `'l'`.The "dtype" of this Record Arrays is given by:  
> ``dtype={'names': ['d','l'],'formats': [('>f2',header['featdim']),'>i2']}``

###Text File (One level header)###

In this type,we use a file containing list of **simple text file** names,each corresponding to single class.It has Following Format:
```
	<feat_dim> <num_classes>
	<data_file1>
	<data_file2>
	..
	..
```
Where `<feat_dim>` is Dimention of input vector and `<num_classes>` is No. of classes, `<data_file1>` is 'simple text file' of class 1,`<data_file2>` is 'simple text file' of class2 and so on.

The **simple text file** has the structure.
```
	<feat_dim> <num_feat_vectors(optional)>
	<feat_vector>
	<feat_vector>
	..
	..
```
Where `<feat_dim>` is Dimention of input vector,`<num_feat_vectors>` is No. of Vectors in this file and
`<feat_vector>` are features of a vector seperated by spaces.Whole file contains only feature vector of single class.

###Text File (Two level header)###

In this type,we use a file containing list of filesname with each with a list of **simple text file** names corresponding to a single class.It has Following Format:
```
	<feat_dim> <num_classes>
	<class_index_file1>
	<class_index_file2>
	..
	..
```
Where `<feat_dim>` is Dimention of input vector and `<num_classes>` is No. of classes, `<class_index_file1>` is 'class index file' (File with list of files of a class) of class 1,`<class_index_file1>` is 'class index file' of class2 and so on.

Each `<class_index_file>` is a file with Following Format:
```
	<td_data_file1>
	<td_data_file2>
	..
	..
```
Each `<td_data_file*>` is name of a  **simple text file**.The **simple text file** has the structure same as that of T1.

------------------------------------------------------------------

*READER TYPE in `data_spec`

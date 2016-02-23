#Data Configuration#


Data Specification has 3 fields:

> 1. `training`
> 2. `validation`
> 3. `testing`

Each one is a json object with following fields:

> * `base_path` :(Mandatory) Base path of data.
> * `filename` :(Mandatory) Filename,
> * `partition` :(Mandatory) Size of data which should be loaded to memory at a time (in MiB)
> * `random` : Whether to use random order (Default value = true)
> * `random_seed` : Seed for random numbers if `random` is `true`
> * `keep_flatten` : Whether to use data as flatten vector or reshape(Default Value = false)
> * `reader_type` : (Mandatory) Type of reader NP/T1/T2.
> * `dim_shuffle` : how to use reshape given fatten vector.Used only `keep_flatten` is `false`

_____________________________________________________________________________________________

**Also See**:

* [Example]({{site.githubUrl}}/tree/master/sample_config/MNIST/CNN/data_spec.json)
* [Reader Type and Data Formats](#data-file-formats)

#!/usr/bin/python
#
# Copyright 2024 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Test-only source fixture from PyPI kaggle 2.0.2.
# Source: https://pypi.org/project/kaggle/2.0.2/
# Modifications: remove type annotations; dedent the two methods for binding to a test double.

def dataset_create_new(self, folder, public=False, quiet=False, convert_to_csv=True, dir_mode='skip'):
    """Creates a new dataset.

        This is similar to creating a new version of a dataset, but it also
        requires additional metadata such as the license and owner.

        Args:
            folder (str): The folder from which to get the metadata file.
            public (bool): Whether the dataset should be public.
            quiet (bool): Suppress verbose output (default is False).
            convert_to_csv (bool): If True, convert data to comma-separated values.
            dir_mode (str): What to do with directories: "skip" - ignore; "zip" - compress and upload.

        Returns:
            ApiCreateDatasetResponse: An ApiCreateDatasetResponse object.
        """
    if not os.path.isdir(folder):
        raise ValueError('Invalid folder: ' + folder)
    meta_file = self.get_dataset_metadata_file(folder)
    with open(meta_file) as f:
        meta_data = json.load(f)
    ref = cast(str, self.get_or_fail(meta_data, 'id'))
    title = cast(str, self.get_or_fail(meta_data, 'title'))
    licenses = cast(List[Dict[str, str]], self.get_or_fail(meta_data, 'licenses'))
    ref_list = ref.split('/')
    owner_slug = ref_list[0]
    dataset_slug = ref_list[1]
    if ref == self.config_values[self.CONFIG_NAME_USER] + '/INSERT_SLUG_HERE':
        raise ValueError('Default slug detected, please change values before uploading')
    if title == 'INSERT_TITLE_HERE':
        raise ValueError('Default title detected, please change values before uploading')
    if len(licenses) != 1:
        raise ValueError('Please specify exactly one license')
    if len(dataset_slug) < 6 or len(dataset_slug) > 50:
        raise ValueError('The dataset slug must be between 6 and 50 characters')
    if len(title) < 6 or len(title) > 50:
        raise ValueError('The dataset title must be between 6 and 50 characters')
    resources = meta_data.get('resources')
    if resources:
        self.validate_resources(folder, resources)
    license_name = self.get_or_fail(licenses[0], 'name')
    description = meta_data.get('description')
    keywords = cast(List[str], self.get_or_default(meta_data, 'keywords', []))
    subtitle = meta_data.get('subtitle')
    if subtitle and (len(subtitle) < 20 or len(subtitle) > 80):
        raise ValueError('Subtitle length must be between 20 and 80 characters')
    found = True
    try:
        self.dataset_status(ref)
    except HTTPError as e:
        found = False
    if found:
        resp = ApiCreateDatasetResponse()
        resp.status = 'error'
        resp.error = f'The requested title "{title}" is already in use by a dataset. Please choose another title.'
        return resp
    request = ApiCreateDatasetRequest()
    request.title = title
    request.slug = dataset_slug
    request.owner_slug = owner_slug
    request.license_name = license_name
    request.subtitle = subtitle
    request.description = description
    request.files = []
    request.is_private = not public
    request.category_ids = keywords
    with ResumableUploadContext() as upload_context:
        self.upload_files(request, resources, folder, ApiBlobType.DATASET, upload_context, quiet, dir_mode)
        with self.build_kaggle_client() as kaggle:
            retry_request = ApiCreateDatasetRequest()
            retry_request.title = title
            retry_request.slug = dataset_slug
            retry_request.owner_slug = owner_slug
            retry_request.license_name = license_name
            retry_request.subtitle = subtitle
            retry_request.description = description
            retry_request.files = request.files
            retry_request.is_private = not public
            retry_request.category_ids = keywords
            response = self.with_retry(kaggle.datasets.dataset_api_client.create_dataset)(retry_request)
            result = cast(ApiCreateDatasetResponse, response)
            if result.error == '':
                result.error = None
            return result

def upload_files(self, request, resources, folder, blob_type, upload_context, quiet=False, dir_mode='skip'):
    """Uploads files in a folder.

        Args:
            request (Union[ApiCreateDatasetVersionRequestBody, ApiCreateModelInstanceRequestBody, ApiCreateDatasetRequest, ApiCreateModelInstanceVersionRequestBody]): The prepared request.
            resources (Optional[List[Dict[str, Union[str, Dict[str, List[Dict[str, str]]]]]]]): The files to upload.
            folder (str): The folder from which to upload.
            blob_type (ApiBlobType): The entity to which the file/blob refers.
            upload_context (ResumableUploadContext): The context for resumable uploads.
            quiet (bool): Suppress verbose output (default is False).
            dir_mode (str): What to do with directories: "skip" - ignore; "zip" - compress and upload.

        Returns:
            None:
        """
    for file_name in os.listdir(folder):
        if file_name in [self.DATASET_METADATA_FILE, self.OLD_DATASET_METADATA_FILE, *self.DATASET_COVER_IMAGE_FILES, self.KERNEL_METADATA_FILE, self.MODEL_METADATA_FILE, self.MODEL_INSTANCE_METADATA_FILE]:
            continue
        upload_file = self._upload_file_or_folder(folder, file_name, blob_type, upload_context, dir_mode, quiet, resources)
        if upload_file is not None:
            files = request.files
            if files is not None:
                files.append(self._new_file(upload_file))

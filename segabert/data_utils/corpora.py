# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""several datasets with preset arguments"""
from .datasets import json_dataset, csv_dataset, json_pos_dataset
import os

class wikipedia(json_dataset):
	"""
	dataset for wikipedia with arguments configured for convenience

	command line usage: `--train-data wikipedia`
	"""
	PATH = 'data/wikipedia/wikidump_lines.json'
	if 'SM_MODEL_DIR' in os.environ:
		PATH = os.path.join(os.environ['SM_CHANNEL_DATA'],PATH)
	assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(wikipedia.PATH), \
                        wikipedia.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)


class wikipedia_pos(json_pos_dataset):
	"""
	dataset for wikipedia_pos with arguments configured for convenience

	command line usage: `--train-data wikipedia_pos`
	"""
	PATH = 'data/wikipedia_pos/wikidump_lines.json'
	if 'SM_MODEL_DIR' in os.environ:
		PATH = os.path.join(os.environ['SM_CHANNEL_DATA'],PATH)
	assert_str = "make sure to set PATH for wikipedia_pos data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(wikipedia_pos.PATH), \
                        wikipedia_pos.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(wikipedia_pos, self).__init__(wikipedia_pos.PATH, **kwargs)


class bookcorpus(json_dataset):
	"""
	dataset for bookcorpus with arguments configured for convenience

	command line usage: `--train-data bookcorpus`
	"""
	PATH = 'data/bookcorpus/bookdump_lines.json'
	if 'SM_MODEL_DIR' in os.environ:
		PATH = os.path.join(os.environ['SM_CHANNEL_DATA'],PATH)
	assert_str = "make sure to set PATH for bookcorpus data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(bookcorpus.PATH), \
                        bookcorpus.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(bookcorpus, self).__init__(bookcorpus.PATH, **kwargs)


class bookcorpus_pos(json_pos_dataset):
	"""
	dataset for bookcorpus_pos with arguments configured for convenience

	command line usage: `--train-data bookcorpus_pos`
	"""
	PATH = 'data/bookcorpus_pos/bookdump_lines.json'
	if 'SM_MODEL_DIR' in os.environ:
		PATH = os.path.join(os.environ['SM_CHANNEL_DATA'],PATH)
	assert_str = "make sure to set PATH for bookcorpus_pos data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(bookcorpus_pos.PATH), \
                        bookcorpus_pos.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(bookcorpus_pos, self).__init__(bookcorpus_pos.PATH, **kwargs)


class webtext(json_dataset):
	"""
	dataset for webtext with arguments configured for convenience

	command line usage: `--train-data webtext`
	"""
	PATH = 'data/webtext/data.json'
	if 'SM_MODEL_DIR' in os.environ:
		PATH = os.path.join(os.environ['SM_CHANNEL_DATA'],PATH)
	assert_str = "make sure to set PATH for webtext data_utils/corpora.py"
	def __init__(self, **kwargs):
		assert os.path.exists(webtext.PATH), \
                        webtext.assert_str
		if not kwargs:
			kwargs = {}
		kwargs['text_key'] = 'text'
		kwargs['loose_json'] = True
		super(webtext, self).__init__(webtext.PATH, **kwargs)


NAMED_CORPORA = {
	'wikipedia': wikipedia,
    'webtext': webtext,
	'wikipedia_pos': wikipedia_pos,
	'bookcorpus':bookcorpus,
	'bookcorpus_pos':bookcorpus_pos,
}

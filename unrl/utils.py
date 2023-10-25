#  Copyright 2023 The unRL Authors.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import unrl.types as t

from functools import wraps


def persisted_generator_value(fn: t.Callable[..., t.Generator[..., ..., t.Any]]) -> t.Callable:
    class PersistentValueGenerator:
        def __init__(self, generator: t.Generator):
            self.gen = generator
            self.value = None

        def __iter__(self):
            return iter(self.gen)

        def send(self, input_: t.Any) -> t.Any:
            try:
                return self.gen.send(input_)
            except StopIteration as response:
                self.gen.close()
                self.value = response.value

    @wraps(fn)
    def decorated(*args, **kwargs) -> PersistentValueGenerator:
        return PersistentValueGenerator(fn(*args, **kwargs))

    return decorated

# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                             |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------- | -------: | -------: | ------: | --------: |
| src/lymph/\_\_init\_\_.py        |       20 |        7 |     65% |     43-49 |
| src/lymph/\_version.py           |       13 |        3 |     77% |      8-11 |
| src/lymph/diagnosis\_times.py    |      247 |       41 |     83% |85, 140, 153, 166, 175-176, 187, 243-244, 277, 334, 346, 350, 358, 367, 370, 374-378, 410, 429-434, 441-448, 452-457, 461-470 |
| src/lymph/graph.py               |      313 |       49 |     84% |56-57, 81, 104, 120-123, 140, 166, 186, 198-214, 265, 276, 279, 296, 346, 349, 352, 365, 371, 555, 568, 634-638, 703-707, 716-721, 741-743 |
| src/lymph/matrix.py              |       76 |       11 |     86% |169, 177-178, 217-218, 234-248 |
| src/lymph/modalities.py          |      146 |       19 |     87% |29, 45, 51, 67, 70, 83, 86, 106, 120, 123, 126, 129, 132, 195, 215-216, 244, 293-294 |
| src/lymph/models/\_\_init\_\_.py |        5 |        0 |    100% |           |
| src/lymph/models/bilateral.py    |      208 |       33 |     84% |129-131, 141-143, 149, 157, 180, 188, 434-440, 460-463, 475-476, 483-489, 501, 548-549, 553-556, 634, 701, 704-705 |
| src/lymph/models/hpv.py          |      148 |      102 |     31% |25-29, 74-93, 103-117, 127-129, 139-141, 146-149, 154-157, 165-184, 197-202, 213-219, 236-242, 246-263, 267-278, 282-283, 287-288, 302-313, 321-330, 334-343, 372-384, 397, 410, 418, 426 |
| src/lymph/models/midline.py      |      343 |       70 |     80% |104, 123, 170, 190, 193, 202, 208, 216, 222, 240, 252, 277-280, 285, 312, 318, 338-341, 421-430, 459-469, 528-530, 535, 537, 572-574, 619, 631, 654-670, 708, 715-718, 752-753, 758, 803, 838, 850-858, 922, 925-926, 929, 955 |
| src/lymph/models/unilateral.py   |      259 |       39 |     85% |111, 125, 135, 163-164, 172, 295-304, 320-334, 529-534, 540-545, 581, 652, 691, 731, 794, 824, 941-942 |
| src/lymph/types.py               |      131 |       14 |     89% |70, 152, 164, 295, 308, 324, 334, 392, 408, 419, 433, 450, 468, 479 |
| src/lymph/utils.py               |      163 |       28 |     83% |20, 24, 29, 34-38, 101-103, 118-120, 219, 225-232, 240-242, 246-247, 459 |
|                        **TOTAL** | **2072** |  **416** | **80%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/lycosystem/lymph/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lycosystem/lymph/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Flycosystem%2Flymph%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/lycosystem/lymph/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
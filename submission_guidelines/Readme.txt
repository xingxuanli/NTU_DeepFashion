We host the test set on CodaLab. Please following the guidelines to ensure your results will be recorded:

1. Register CodaLAB with your NTU email, with the last 5 digits of your matric number as your username.
2. Submit a zip file containing a single file “prediction.txt” with 1000 rows of your prediction results on the test set. For example, the first 5 line should be like:

5 2 0 3 2 2
5 2 1 3 2 2
5 0 2 0 2 2
5 0 2 0 5 2
1 2 2 3 2 2
… (1000 lines in total)

3. Note that your predictions should follow the 1000 test set images’ sequential order in the test.txt, which means you should set “shuffle=False” here if you use PyTorch DataLoader in your code.

Resubmission is allowed. Please report your best score in your report.

Submit the following files to NTU Learn before the deadline:
Your pdf technical report (in CVPR format). Please name it as
[YOUR_NAME]_[MATRIC_NO]_[project_1].pdf

A screenshot from the CodaLab leaderboard, with your username and best score. We will use the score from CodaLab for marking, but will keep your screenshot here for double-check reference.

! For submission format, you can refer to prediction.zip in the same folder.
If you submit the sampled prediction.zip correctly, your score will be "0.4541666667". Note that this is just for platform test use.

You should name your test_attr.txt as prediction.txt, and zip it.
Submit your zip file to Codalab and wait for the STATUS to be "Finished". You can choose whether to show your score in the leaderboard.


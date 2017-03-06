import os
temp1 = "temp1.m4a"
temp2 = "temp2.m4a"
temp3 = "temp3.m4a"
all_files = [x for x in os.listdir("input/")]
for file in all_files:
  if file.endswith(".m4a"):
    print os.popen("ffmpeg -i long/" + file + " -af silenceremove=1:0:-96dB " + temp1).read()
    print os.popen("ffmpeg -i " + temp1 + " -af areverse " + temp2).read()
    print os.popen("ffmpeg -i " + temp2 + " -af silenceremove=1:0:-96dB " + temp3).read()
    print os.popen("ffmpeg -i " + temp3 + " -af areverse output/" + file).read()
    os.remove(temp1)
    os.remove(temp2)
    os.remove(temp3)

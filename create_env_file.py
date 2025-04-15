import os

path=os.getcwd()

with open("AI_HEAD_ENV.sh","w+") as f:

  f.write(f"export AI_HEAD_PATH={path}\n")
  f.write(f"export AI_HEAD_LIB_PATH={path}/utils\n")
  f.write(f"export PATH={path}/:$PATH\n")
import os
import glob
import cpphint

root_path = path = os.getcwd()
include_dir = os.path.join(root_path, "include")
src_dir = os.path.join(root_path, "src")
tools_dir = os.path.join(root_path, "tools")
check_dirs = [include_dir, src_dir, tools_dir]
cpp_files = []
h_files = []
for root in check_dirs:
    cpp_files += glob.glob(root + "/**/*.cpp", recursive=True)
    h_files += glob.glob(root + "/**/*.h", recursive=True)


def cpplint():
    cpplint_filters = [
        # "-build/include_order",
        # "-build/namespaces",
        # "-whitespace/comments",
        # "-whitespace/indent",
    ]
    command = "cpplint "
    if len(cpplint_filters) != 0:
        command += "--filter="
        for filter in cpplint_filters:
            command += "{},".format(filter)
    for file in cpp_files:
        os.system("{} {}".format(command, file))
    for file in h_files:
        os.system("{} {}".format(command, file))


if __name__ == "__main__":
    cpplint()

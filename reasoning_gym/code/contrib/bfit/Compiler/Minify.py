def minify(code):
    old_code = ""

    while old_code != code:
        old_code = code

        code = code.replace("><", "")
        code = code.replace("<>", "")
        code = code.replace("+-", "")
        code = code.replace("-+", "")

        code = code.replace("][-]", "]")

    return code

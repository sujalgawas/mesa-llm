BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

# Color mapping for easy access
COLORS = {
    "purple": "\033[35m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "orange": "\033[38;5;208m",
    "pink": "\033[38;5;205m",
}


def print_styled(*args, color=None, bold=False, underline=False, **kwargs):
    """
    Print text with optional styling.

    Args:
        *args: Text to print (same as regular print)
        color (str, optional): Color name (purple, cyan, green, red, yellow, blue, magenta, orange, pink)
        bold (bool): Whether to make text bold
        underline (bool): Whether to underline text
        **kwargs: Additional keyword arguments passed to print()

    Example:
        print_styled("Hello World!", color="pink", bold=True)
        print_styled("Warning message", color="red", underline=True)
        print_styled("Success!", color="green", bold=True)
    """
    # Build style string
    style_codes = []

    # Add color if specified
    if color and color.lower() in COLORS:
        style_codes.append(COLORS[color.lower()])

    # Add styles if specified
    if bold:
        style_codes.append(BOLD)
    if underline:
        style_codes.append(UNDERLINE)

    # Combine all style codes
    style_prefix = "".join(style_codes)

    # Convert args to strings and apply styling
    if style_prefix:
        styled_args = []
        for arg in args:
            styled_args.append(f"{style_prefix}{arg}{RESET}")
        print(*styled_args, **kwargs)
    else:
        # No styling, just regular print
        print(*args, **kwargs)


def style_txt(text, color=None, bold=False, underline=False):
    """
    Return styled text string without printing.

    Args:
        text (str): Text to style
        color (str, optional): Color name
        bold (bool): Whether to make text bold
        underline (bool): Whether to underline text

    Returns:
        str: Styled text with ANSI codes

    Example:
        styled_text = style_txt("Hello", color="pink", bold=True)
        print(styled_text)
    """
    style_codes = []

    if color and color.lower() in COLORS:
        style_codes.append(COLORS[color.lower()])

    if bold:
        style_codes.append(BOLD)
    if underline:
        style_codes.append(UNDERLINE)

    style_prefix = "".join(style_codes)

    if style_prefix:
        return f"{style_prefix}{text}{RESET}"
    else:
        return text

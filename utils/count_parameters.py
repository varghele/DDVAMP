def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a neural network model.

    Args:
        model: The neural network model to analyze

    Returns:
        int: Total number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Format the number for better readability
    formatted_count = f"{total_params:,}"
    print(f"Total trainable parameters: {formatted_count}")

    return total_params
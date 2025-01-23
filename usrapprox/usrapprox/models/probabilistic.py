# from math import pi
# import torch

# # from usrapprox.usrapprox.utils.utils import Categories


# def calculate_logits(n: torch.Tensor, linear=False) -> torch.Tensor:

#     if linear:
#         listen_logit = torch.full_like(n, 0.5)  # listen: more likely as n -> 1
#         skip_logit = torch.zeros_like(n)  # skip: more likely as n -> 0
#         like_logit = 0.6 * n  # like: more likely as n -> 1
#         dislike_logit = 0.6 * (1 - n)  # dislike: more likely as n -> 0
#         love_logit = 0.9 * n  # loved: more likely as n -> 1
#         hate_logit = 0.9 * (1 - n)  # hated: more likely as n -> 0
#     else:
#         mu, sigma = 0.4, 0.4

#         listen_logit = (
#             1
#             / torch.sqrt(torch.tensor(2 * pi * sigma**2))
#             * torch.exp(-4 * (n - mu - 0.2) ** 2 / sigma**2)
#             * 20
#             / 31
#         )
#         skip_logit = (
#             1
#             / torch.sqrt(torch.tensor(2 * pi * sigma**2))
#             * torch.exp(-4 * (n - mu) ** 2 / sigma**2)
#             * 20
#             / 31
#         )
#         like_logit = (1 / (1 + torch.exp(-10 * (n - 0.6)))) * 0.8
#         dislike_logit = (1 - 1 / (1 + torch.exp(-10 * (n - (1 - 0.6))))) * 0.8
#         love_logit = 1 / (1 + torch.exp(-(1.4444 + 10 * (n - 0.85))))
#         hate_logit = 1 - 1 / (1 + torch.exp(0.44439 - 10 * (n - 0.25)))

#     result = torch.stack(
#         (hate_logit, dislike_logit, skip_logit, listen_logit, like_logit, love_logit),
#         dim=-1,
#     )
#     return result


# def probabilistic_model_torch(input_values: torch.Tensor, linear=False) -> torch.Tensor:
#     raise NotImplementedError("Deprecated.")
#     """
#     Probabilistic 'model' that maps input values to feedback categories.

#     Note that the output values are not probabilities, but feedback categories.
#     Their values can be mapped as:
#     - hated: 1
#     - dislike: 2
#     - skip: 3
#     - listen: 4
#     - like: 5
#     - loved: 6

#     # Which is the class Categories.

#     Args:
#         input_values (torch.Tensor): Input values between 0 and 1.

#     Returns:
#         torch.Tensor: Feedback categories for each input value
#     """
#     if isinstance(input_values, list):
#         input_values = torch.tensor(input_values, dtype=torch.float32)

#     if not torch.all((0.0 <= input_values) & (input_values <= 1.0)):
#         raise ValueError("Input values must be in range [0,1].")

#     logits = calculate_logits(input_values, linear)
#     probabilities = logits / torch.sum(logits, dim=-1, keepdim=True)

#     distribution = torch.distributions.Categorical(probs=probabilities)

#     samples = distribution.sample()
#     samples = samples / (probabilities.shape[-1] - 1) * 2 - 1

#     return samples

#     # import matplotlib.pyplot as plt
#     # n = torch.tensor([0.1, 0.5, 0.9])
#     # linear = True
#     # result = calculate_logits(n, linear)
#     # print(result)

#     # # set torch as deterministic
#     # torch.manual_seed(0)
#     # torch.use_deterministic_algorithms(True)

#     # input_values = torch.linspace(0, 1, 100)  # 100 points between 0 and 1

#     # p = calculate_logits(input_values).transpose(0, 1)

#     # # Plot probabilities
#     # plt.figure(figsize=(10, 6))
#     # for category, values in zip(Categories, p):
#     #     plt.plot(input_values.numpy(), values, label=category, linewidth=2)

#     # # Add labels, legend, and grid
#     # plt.title("Mapping of Input Values to Probabilities", fontsize=16)
#     # plt.xlabel("Input Value", fontsize=14)
#     # plt.ylabel("Probability", fontsize=14)
#     # plt.legend(title="Categories", fontsize=12)
#     # plt.grid(alpha=0.3)

#     # # Test probabilistic model and visualize results
#     # input_values = torch.linspace(0, 1, 100)  # 100 points between 0 and 1
#     # probs = probabilistic_model_torch(input_values)

#     # # Plot probabilistic model outputs
#     # plt.figure(figsize=(10, 6))
#     # plt.plot(
#     #     input_values.numpy(), probs.numpy(), "ro", label="Category Outputs", linewidth=2
#     # )

#     # # # Add labels, legend, and grid
#     # plt.title("Mapping of Input Values to Probabilities", fontsize=16)
#     # plt.xlabel("Input Value", fontsize=14)
#     # plt.ylabel("Probability", fontsize=14)
#     # plt.legend(title="Categories", fontsize=12)
#     # plt.grid(alpha=0.3)

#     # # Plot probabilities distribution
#     # plt.hist(
#     #     probs.numpy(), bins=len(Categories), color="blue", edgecolor="black", alpha=0.7
#     # )
#     # plt.title("Probabilities Distribution")
#     # plt.xlabel("Categories")
#     # plt.ylabel("Frequency")
#     # plt.show()

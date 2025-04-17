from jax import numpy as jnp
from jax import lax
import jax

def likelihood_ratio_loss(model_outputs, labels, loss_type_code=0, weights=None):
    """
    Unified loss function for likelihood ratio estimation
    This function roughly follows the parameterization in:
        - Learning Likelihood Ratios with Neural Network Classifiers
        - https://arxiv.org/pdf/2305.10500v2

    Args:
        model_outputs: Raw model outputs (before activation).
        labels: Ground truth labels (0 for p(x|θ1), 1 for p(x|θ0)).
        loss_type_code: integer code for the loss type
        weights: Optional sample weights (default is uniform).
    
    Returns:
        Scalar loss value.
    """
    model_outputs = jnp.squeeze(model_outputs)
    labels = jnp.squeeze(labels)

    # We later clip the loss type labels so if out of bounds then it will revert to the limiting cases
    preds = lax.cond(
        loss_type_code < 2,
        lambda _: jnp.clip(jax.nn.sigmoid(model_outputs), 1e-7, 1 - 1e-7),  # Maps to (0,1)
        lambda _: jnp.clip(jnp.exp(model_outputs), 1e-7, 1e7),  # Maps to (0,∞) with clipping to prevent explosions
        operand=None
    )
    
    # These definitions are from Table 1 in https://arxiv.org/pdf/2305.10500v2
    #   Compute loss terms A(f) and B(f) in a jax safe way
    def loss_0(_): # BCE where A(f) = log(f) and B(f) = log(1 - f)
        return jnp.log(preds), jnp.log(1 - preds)

    def loss_1(_): # MSE where A(f) = -(1 - f)^2 and B(f) = f^2
        return -(1 - preds) ** 2, -(preds ** 2)

    def loss_2(_): # MLC where A(f) = log(f) and B(f) = 1 - f
        return jnp.log(preds), 1 - preds

    def loss_3(_): # SQRT where A(f) = -1/sqrt(f) and B(f) = -sqrt(f)
        return -1 / jnp.sqrt(preds), -jnp.sqrt(preds)

    pos_loss, neg_loss = lax.switch(
        jnp.clip(loss_type_code, 0, 3),  # Ensure loss_type_code is in valid range
        [loss_0, loss_1, loss_2, loss_3],
        operand=None
    )

    # Assign loss based on labels
    loss = -1 * (labels * pos_loss + (1 - labels) * neg_loss)

    # Apply sample weights if provided
    if weights is None:
        return jnp.mean(loss)
    else:
        weights = jnp.squeeze(weights)
        return jnp.mean(weights * loss)

def gradient_regularization_loss(model, x, transition_sensitivity=0.5):
    """
    Compute adaptive gradient regularization loss with reduced penalty in high-gradient regions.
    
    Args:
        model: neural network model
        x: Input data
        transition_sensitivity: exponential suppression of gradients
    
    Returns:
        Adaptive regularization loss
    """
    
    def model_fn(inputs):
        return jnp.sum(model(inputs))
    
    # Compute L2 norm of gradients with clipping
    batch_gradients = jax.grad(model_fn)(x)    
    batch_gradients = jnp.clip(batch_gradients, -1e3, 1e3)
    gradient_norms  = jnp.sum(batch_gradients**2, axis=1)
    
    # Apply adaptive weighting that reduces penalty for high gradients
    suppression_factor = jnp.exp(-transition_sensitivity * jnp.sqrt(gradient_norms))
    
    loss = jnp.mean(suppression_factor * gradient_norms)
    
    return loss

def likelihood_ratio_grad_regularized_loss_fn(model, batch, loss_type_code=0, reg_strength=0.0, transition_sensitivity=0.5):
    """Combined loss function with main loss and gradient regularization."""
    x, y, weights = batch['x'], batch['y'], batch.get('weights', None)
    
    # Get model outputs
    logits = model(x)
    
    # Compute main loss
    main_loss = likelihood_ratio_loss(logits, y, loss_type_code, weights)
    
    # Compute gradient regularization if needed
    # NOTE: Need to do this in eval mode to not use dropout (if dropout is needed, RNG keys must be provided I think)
    model.eval()
    grad_loss = lax.cond(
        reg_strength > 0, 
        lambda _: gradient_regularization_loss(model, x, transition_sensitivity), 
        lambda _: 0.0, 
        operand=None)
    model.train()
        
    # Combine losses
    total_loss = main_loss + reg_strength * grad_loss
    
    return total_loss, (logits, main_loss, grad_loss)

def convert_to_probabilities(model_outputs, loss_type_code):
    """Convert model outputs to probabilities based on loss type."""
    preds = lax.cond(
        loss_type_code < 2,
        lambda _: jax.nn.sigmoid(model_outputs),
        lambda _: jnp.exp(model_outputs),
        operand=None
    )
    
    probs = lax.cond(
        loss_type_code < 2,
        lambda _: preds,
        lambda _: preds / (1 + preds),  # convert to probability space
        operand=None
    )
    
    return probs
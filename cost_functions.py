import jax.numpy as jnp
import numpy as np
import jax
import functools
from jax import grad, jacfwd, vmap

# ================================================================
# Cost-Sensitive XGBoost
# ================================================================

def cost_sensitive_instance_loss(predictions, label, cost_negative=1.0, cost_positive=1.0):
    prob = jax.nn.sigmoid(predictions)
    prob = jnp.clip(prob, 1e-7, 1 - 1e-7)
    return -(
        label * cost_positive * jnp.log(prob) +
        (1 - label) * cost_negative * jnp.log(1 - prob)
    )

def create_cost_sensitive_objective(cost_negative=1.0, cost_positive=1.0):
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        preds_jnp = jnp.array(preds)
        labels_jnp = jnp.array(labels)
        grad_fn = vmap(grad(lambda p, y: cost_sensitive_instance_loss(
            p, y, cost_negative=cost_negative, cost_positive=cost_positive)))
        hess_fn = vmap(jacfwd(grad(lambda p, y: cost_sensitive_instance_loss(
            p, y, cost_negative=cost_negative, cost_positive=cost_positive))))
        grads = np.array(grad_fn(preds_jnp, labels_jnp))
        hess = np.array(hess_fn(preds_jnp, labels_jnp))
        return grads, hess
    return objective

# ================================================================
# TF Focal Loss
# ================================================================


def _binary_focal_loss_from_logits(labels, logits, gamma, pos_weight, label_smoothing):
    def _process_labels(labels, label_smoothing, dtype):
        labels = np.array(labels, dtype=dtype)
        if label_smoothing is not None:
            labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        return labels

    labels = _process_labels(labels, label_smoothing, logits.dtype)
    p = 1 / (1 + np.exp(-logits))

    if label_smoothing is None:
        labels_shape = labels.shape
        logits_shape = logits.shape
        if labels_shape != logits_shape:
            shape = np.broadcast_shapes(labels_shape, logits_shape)
            labels = np.broadcast_to(labels, shape)
            logits = np.broadcast_to(logits, shape)
        
        if pos_weight is None:
            def loss_func(labels, logits):
                return np.log1p(np.exp(-np.abs(logits))) + np.maximum(-logits, 0) * (labels - (logits < 0))
        else:
            def loss_func(labels, logits, pos_weight=pos_weight):
                targets = labels * pos_weight
                return np.log1p(np.exp(-np.abs(logits))) + np.maximum(-logits, 0) * (targets - (logits < 0))
        
        loss = loss_func(labels, logits)
        modulation_pos = (1 - p) ** gamma
        modulation_neg = p ** gamma
        mask = labels.astype(bool)
        modulation = np.where(mask, modulation_pos, modulation_neg)
        return modulation * loss

    pos_term = labels * ((1 - p) ** gamma)
    neg_term = (1 - labels) * (p ** gamma)
    log_weight = pos_term
    if pos_weight is not None:
        log_weight *= pos_weight
    log_weight += neg_term
    log_term = np.log1p(np.exp(-np.abs(logits)))
    log_term += np.maximum(-logits, 0)
    log_term *= log_weight
    loss = neg_term * logits + log_term
    return loss

def tf_focal_loss_objective(gamma=1.0, pos_weight=1.0, label_smoothing=None):
    """Create XGBoost objective function for TensorFlow focal loss"""
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        preds_jnp = jnp.array(preds)
        labels_jnp = jnp.array(labels)
        grad_fn = vmap(grad(lambda p, y: _binary_focal_loss_from_logits(
            p, y, gamma=gamma, pos_weight=pos_weight, label_smoothing=label_smoothing)))
        hess_fn = vmap(jacfwd(grad(lambda p, y: cost_sensitive_instance_loss(
            p, y, gamma=gamma, pos_weight=pos_weight, label_smoothing=label_smoothing))))
        grads = np.array(grad_fn(preds_jnp, labels_jnp))
        hess = np.array(hess_fn(preds_jnp, labels_jnp))
        return grads, hess
    return objective


# ================================================================
# Focal Loss with Cost Sensitivity
# ================================================================

def focal_cost_loss(predictions, labels, cost_negative=1.0, cost_positive=1.0, 
                   focusing_param=2.0, class_weight=1.0):
    """Focal loss with cost-sensitive weighting"""
    prob = jax.nn.sigmoid(predictions)
    prob = jnp.clip(prob, 1e-7, 1 - 1e-7)
    
    return -(labels * class_weight * (1 - prob) ** focusing_param * cost_positive * jnp.log(prob) + (1 - labels) * (prob) ** focusing_param * cost_negative * jnp.log(1 - prob))
    
    

def create_focal_cost_objective(cost_negative=1.0, cost_positive=1.0, 
                               focusing_param=2.0, class_weight=1.0):
    """Create XGBoost objective function for focal loss with cost sensitivity"""
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        preds_jnp = jnp.array(preds)
        labels_jnp = jnp.array(labels)
        grad_fn = vmap(grad(lambda p, y: cost_sensitive_instance_loss(
            p, y, cost_negative=cost_negative, cost_positive=cost_positive, focusing_param=focusing_param, class_weight=class_weight))),
        hess_fn = vmap(jacfwd(grad(lambda p, y: cost_sensitive_instance_loss(
            p, y, cost_negative=cost_negative, cost_positive=cost_positive, focusing_param=focusing_param, class_weight=class_weight))))
        grads = np.array(grad_fn(preds_jnp, labels_jnp))
        hess = np.array(hess_fn(preds_jnp, labels_jnp))
        return grads, hess
    return objective

# ================================================================
# GEV Link Focal Loss
# ================================================================

def gev_focal_loss(predictions, labels, shape_param=-0.25, 
                  focusing_param=2.0, class_weight=1.0):
    """Focal loss with Generalized Extreme Value (GEV) link function"""
    # GEV link function (maps to [0,1])
    linear_term = 1 + shape_param * predictions
    probs = jnp.exp(-jnp.power(linear_term, -1/shape_param))
    
    # Focal loss weights
    pos_weight = class_weight * jnp.power(1 - probs, focusing_param)
    neg_weight = jnp.power(probs, focusing_param)
    
    
    
    return -(labels * class_weight * (1 - prob) ** focusing_param * cost_positive * jnp.log(prob) + (1 - labels) * (prob) ** focusing_param * cost_negative * jnp.log(1 - prob))
    
    return loss

def create_gev_focal_objective(shape_param=-0.25, focusing_param=2.0, class_weight=1.0):
    """Create XGBoost objective function for GEV focal loss"""
    loss_fn = functools.partial(
        gev_focal_loss,
        shape_param=shape_param,
        focusing_param=focusing_param,
        class_weight=class_weight
    )
    
    def objective(predictions, labels):
        gradient_fn = grad(lambda p: loss_fn(p, labels))
        hessian_fn = hessian(lambda p: loss_fn(p, labels))
        
        gradients = gradient_fn(predictions)
        hessians = jnp.diag(hessian_fn(predictions))
        
        return gradients, hessians
    
    return objective

# ================================================================
# Unified Focal Loss (Focal + Tversky)
# ================================================================

def unified_focal_loss(predictions, labels, class_weight=1.0, tversky_beta=0.7,
                      focal_gamma=2.0, tversky_gamma=1.0, balance_weight=0.5):
    """Unified focal loss combining focal and Tversky components"""
    probs = jnp.clip(predictions, 1e-7, 1 - 1e-7)
    
    # Focal loss component
    pos_weight = class_weight * jnp.power(1 - probs, focal_gamma)
    neg_weight = jnp.power(probs, focal_gamma)
    
    focal_loss = -jnp.mean(
        labels * pos_weight * jnp.log(probs + 1e-7) +
        (1 - labels) * neg_weight * jnp.log(1 - probs + 1e-7)
    )
    
    # Tversky loss component
    true_pos = jnp.sum(labels * probs)
    false_pos = jnp.sum((1 - labels) * probs)
    false_neg = jnp.sum(labels * (1 - probs))
    
    tversky_index = true_pos / (true_pos + tversky_beta * false_neg + 
                               (1 - tversky_beta) * false_pos + 1e-7)
    tversky_loss = jnp.power(1 - tversky_index, tversky_gamma)
    
    # Combine losses
    total_loss = balance_weight * focal_loss + (1 - balance_weight) * tversky_loss
    return total_loss

def create_unified_focal_objective(class_weight=1.0, tversky_beta=0.7,
                                  focal_gamma=2.0, tversky_gamma=1.0, 
                                  balance_weight=0.5):
    """Create XGBoost objective function for unified focal loss"""
    loss_fn = functools.partial(
        unified_focal_loss,
        class_weight=class_weight,
        tversky_beta=tversky_beta,
        focal_gamma=focal_gamma,
        tversky_gamma=tversky_gamma,
        balance_weight=balance_weight
    )
    
    def objective(predictions, labels):
        gradient_fn = grad(lambda p: loss_fn(p, labels))
        hessian_fn = hessian(lambda p: loss_fn(p, labels))
        
        gradients = gradient_fn(predictions)
        hessians = jnp.diag(hessian_fn(predictions))
        
        return gradients, hessians
    
    return objective


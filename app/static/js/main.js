// Main JavaScript file for Hotel Booking Predictor

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add animation classes to elements as they come into view
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);

    // Observe cards and other elements for animation
    document.querySelectorAll('.card, .how-it-works, .cta-section').forEach(el => {
        observer.observe(el);
    });
});

// Form validation and enhancement
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                
                // Focus on first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.focus();
                }
            }
            
            form.classList.add('was-validated');
        });
    });
}

// Initialize form validation when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeFormValidation);

// Dynamic form calculations
function initializeDynamicCalculations() {
    const weekendNights = document.getElementById('number_of_weekend_nights');
    const weekNights = document.getElementById('number_of_week_nights');
    const adults = document.getElementById('number_of_adults');
    const children = document.getElementById('number_of_children');
    const price = document.getElementById('average_price');
    
    function updateCalculations() {
        const totalNights = parseInt(weekendNights?.value || 0) + parseInt(weekNights?.value || 0);
        const totalGuests = parseInt(adults?.value || 0) + parseInt(children?.value || 0);
        const avgPrice = parseFloat(price?.value || 0);
        
        // Update any display elements if they exist
        const totalNightsDisplay = document.getElementById('total-nights-display');
        const totalGuestsDisplay = document.getElementById('total-guests-display');
        const totalCostDisplay = document.getElementById('total-cost-display');
        
        if (totalNightsDisplay) totalNightsDisplay.textContent = totalNights;
        if (totalGuestsDisplay) totalGuestsDisplay.textContent = totalGuests;
        if (totalCostDisplay) totalCostDisplay.textContent = `$${(totalNights * avgPrice).toFixed(2)}`;
        
        // Validate minimum requirements
        if (totalNights === 0 && (weekendNights || weekNights)) {
            showValidationMessage('At least one night (weekend or week) is required.', 'warning');
        }
        
        if (totalGuests === 0 && adults) {
            showValidationMessage('At least one adult is required.', 'warning');
        }
    }
    
    // Add event listeners for real-time calculations
    [weekendNights, weekNights, adults, children, price].forEach(input => {
        if (input) {
            input.addEventListener('input', updateCalculations);
        }
    });
    
    // Initial calculation
    updateCalculations();
}

// Initialize dynamic calculations when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeDynamicCalculations);

// Show validation messages
function showValidationMessage(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.dynamic-alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show dynamic-alert`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the form
    const form = document.querySelector('form');
    if (form) {
        form.insertBefore(alertDiv, form.firstChild);
    }
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Loading states
function showLoading(buttonElement, text = 'Processing...') {
    if (buttonElement) {
        buttonElement.disabled = true;
        buttonElement.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            ${text}
        `;
    }
}

function hideLoading(buttonElement, originalText) {
    if (buttonElement) {
        buttonElement.disabled = false;
        buttonElement.innerHTML = originalText;
    }
}

// API request helper
async function makeAPIRequest(url, data, method = 'POST') {
    try {
        const response = await fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Copy to clipboard functionality
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showValidationMessage('Copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy to clipboard:', err);
        showValidationMessage('Failed to copy to clipboard.', 'danger');
    });
}

// Print functionality
function printResults() {
    window.print();
}

// Download results as text
function downloadResults(resultData, filename = 'prediction_result.txt') {
    const text = `
Hotel Booking Prediction Result
Generated on: ${new Date().toLocaleString()}

Prediction: ${resultData.prediction}
Confidence: ${resultData.confidence}%
Model: ${resultData.model_name}

Input Data:
${JSON.stringify(resultData.input_data, null, 2)}
    `.trim();
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

// Theme switcher (if implemented)
function toggleTheme() {
    const body = document.body;
    const isDark = body.classList.contains('dark-theme');
    
    if (isDark) {
        body.classList.remove('dark-theme');
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    }
}

// Load saved theme
function loadSavedTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
}

// Initialize theme on load
document.addEventListener('DOMContentLoaded', loadSavedTheme);

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.querySelector('form');
        if (form) {
            form.requestSubmit();
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const modalInstance = bootstrap.Modal.getInstance(modal);
            if (modalInstance) {
                modalInstance.hide();
            }
        });
    }
});

// Analytics and tracking (placeholder)
function trackEvent(eventName, eventData = {}) {
    // Implement analytics tracking here
    console.log('Event tracked:', eventName, eventData);
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // Could send error to monitoring service
});

// Performance monitoring
window.addEventListener('load', function() {
    const loadTime = performance.now();
    console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    trackEvent('page_load_time', { loadTime });
});

// Export functions for global use
window.HotelBookingPredictor = {
    showLoading,
    hideLoading,
    makeAPIRequest,
    copyToClipboard,
    printResults,
    downloadResults,
    toggleTheme,
    trackEvent,
    showValidationMessage
};


// JavaScript for automation dashboard functionality

// Load patterns on page load
document.addEventListener('DOMContentLoaded', function() {
    loadPatterns();
});

async function loadPatterns() {
    try {
        const response = await fetch('/api/load-patterns', {
            method: 'POST'
        });
        const result = await response.json();
        
        if (result.success) {
            console.log(`Loaded ${result.patterns_loaded} patterns`);
            // Refresh page to show loaded patterns
            setTimeout(() => window.location.reload(), 1000);
        }
    } catch (error) {
        console.error('Error loading patterns:', error);
    }
}

async function generateAutomation(patternName) {
    const button = event.target;
    const originalText = button.innerHTML;
    
    // Show loading state
    button.innerHTML = '<span class="loading-spinner"></span>Generating...';
    button.disabled = true;
    
    try {
        const response = await fetch('/api/generate-automation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                pattern_id: patternName
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Redirect to setup page
            window.location.href = `/automation/${result.automation_id}`;
        } else {
            alert('Error generating automation: ' + result.error);
            button.innerHTML = originalText;
            button.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error. Please try again.');
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

async function installAutomation(automationId) {
    const button = event.target;
    const originalText = button.innerHTML;
    
    button.innerHTML = '<span class="loading-spinner"></span>Installing...';
    button.disabled = true;
    
    try {
        const response = await fetch(`/api/install-automation/${automationId}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Start polling for installation progress
            pollInstallationProgress(automationId);
        } else {
            alert('Installation failed: ' + result.error);
            button.innerHTML = originalText;
            button.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error. Please try again.');
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

async function pollInstallationProgress(automationId) {
    const progressElement = document.getElementById('installation-progress');
    const installButton = document.querySelector('.install-btn');
    
    const poll = async () => {
        try {
            const response = await fetch(`/api/automation-status/${automationId}`);
            const result = await response.json();
            
            if (result.success) {
                const automation = result.automation;
                
                if (progressElement) {
                    progressElement.textContent = automation.progress;
                }
                
                if (automation.status === 'installed') {
                    // Installation complete
                    progressElement.innerHTML = '✅ Installation complete!';
                    installButton.style.display = 'none';
                    
                    // Show next step
                    const nextStep = document.getElementById('credentials-step');
                    if (nextStep) {
                        nextStep.style.display = 'block';
                    }
                    
                    return; // Stop polling
                } else if (automation.status === 'install_failed') {
                    progressElement.innerHTML = '❌ Installation failed: ' + automation.error;
                    installButton.innerHTML = 'Retry Installation';
                    installButton.disabled = false;
                    return; // Stop polling
                }
                
                // Continue polling
                setTimeout(poll, 2000);
            }
        } catch (error) {
            console.error('Error polling status:', error);
            setTimeout(poll, 5000); // Retry after longer delay
        }
    };
    
    poll();
}

async function saveCredentials(automationId) {
    const form = document.getElementById('credentials-form');
    const formData = new FormData(form);
    const button = document.querySelector('.credentials-btn');
    const originalText = button.innerHTML;
    
    button.innerHTML = '<span class="loading-spinner"></span>Testing...';
    button.disabled = true;
    
    try {
        const response = await fetch(`/api/save-credentials/${automationId}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Show success message
            showMessage('✅ Credentials validated successfully!', 'success');
            
            // Show activation step
            const activationStep = document.getElementById('activation-step');
            if (activationStep) {
                activationStep.style.display = 'block';
            }
            
            button.innerHTML = '✅ Validated';
            button.disabled = true;
        } else {
            showMessage('❌ ' + result.error, 'error');
            button.innerHTML = originalText;
            button.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage('❌ Network error. Please try again.', 'error');
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

async function activateAutomation(automationId) {
    const button = event.target;
    const originalText = button.innerHTML;
    
    button.innerHTML = '<span class="loading-spinner"></span>Activating...';
    button.disabled = true;
    
    try {
        const response = await fetch(`/api/activate-automation/${automationId}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Show success message
            showMessage('🎉 ' + result.message, 'success');
            
            // Show next steps
            if (result.next_steps) {
                let nextStepsHtml = '<h3>What happens next:</h3><ul>';
                result.next_steps.forEach(step => {
                    nextStepsHtml += `<li>${step}</li>`;
                });
                nextStepsHtml += '</ul>';
                
                document.getElementById('success-details').innerHTML = nextStepsHtml;
            }
            
            button.innerHTML = '✅ Activated';
            button.disabled = true;
            
            // Redirect to dashboard after 3 seconds
            setTimeout(() => {
                window.location.href = '/';
            }, 3000);
            
        } else {
            showMessage('❌ ' + result.error, 'error');
            button.innerHTML = originalText;
            button.disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage('❌ Network error. Please try again.', 'error');
        button.innerHTML = originalText;
        button.disabled = false;
    }
}

function showMessage(message, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = type === 'success' ? 'success-message' : 'error-message';
    messageDiv.innerHTML = message;
    
    // Insert at top of main content
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);
    
    // Remove after 5 seconds
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// Auto-refresh automation status every 30 seconds
setInterval(async () => {
    const automationId = window.location.pathname.split('/').pop();
    if (automationId && automationId !== '') {
        try {
            const response = await fetch(`/api/automation-status/${automationId}`);
            const result = await response.json();
            
            if (result.success) {
                const automation = result.automation;
                
                // Update progress if element exists
                const progressElement = document.getElementById('current-progress');
                if (progressElement && automation.progress) {
                    progressElement.textContent = automation.progress;
                }
            }
        } catch (error) {
            // Ignore errors for auto-refresh
        }
    }
}, 30000);

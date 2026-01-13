const alertContainer = document.getElementById('alert-container');

function showAlert(type, message) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show pb-3`;
    alert.role = 'alert';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
    alertContainer.innerHTML = ''; // clear previous
    alertContainer.appendChild(alert);

    setTimeout(() => {
        bootstrap.Alert.getOrCreateInstance(alert).close();
    }, 5000);
}
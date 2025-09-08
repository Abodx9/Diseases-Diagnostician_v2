
    $(document).ready(function() {
      $('#chat-button').on('click', function() {
        $('#chat-container').toggle();
      });
    });
	document.getElementById('camera-icon').addEventListener('click', function() {
  document.getElementById('file-input').click();
});



// Only one model can be used (Selected) at the same time
      $('#resnet').on('change', function() {
        if (this.checked) {
          $('#trnsf').prop('checked', false);
        }
        
        console.log('Resnet 50 model is in use', this.checked);
      });

    
      $('#trnsf').on('change', function() {
        if (this.checked) {
          $('#resnet').prop('checked', false);
        }
      
        console.log('Transformer Vision model is in use', this.checked);
      });
	  
	  

document.getElementById('file-input').addEventListener('change', function() {
  const fileInput = this;  // 'this' refers to the file input element
  if (fileInput.files.length > 0) {
    const uploadedPhoto = fileInput.files[0];
    displayUserPhoto(uploadedPhoto);
    const resnetSwitch = document.getElementById('resnet');
    const transformerSwitch = document.getElementById('trnsf');

      if (resnetSwitch.checked) {
        resnetResponse(uploadedPhoto);
    } 
      else if (transformerSwitch.checked) {
        transResponse(uploadedPhoto);
    }
    else{alert("Please Select a model first!");}

  }
});

    let titl = document.title;
    window.addEventListener("blur", () => {
       document.title = "Where did u go ! :("
         })
    window.addEventListener("focus", () => {
         document.title = titl;
          })

    const form = document.getElementById('chat-form');
    const input = form.querySelector('.user-entry__input');
    const chatWindow = document.getElementById('chat-window');


    // The main Button
    form.addEventListener('submit', (event) => {
      event.preventDefault();
    
      const userMessage = input.value;
     
        getBotResponse(userMessage);
      
    
      // Clear the input field, so the user can use it again
      input.value = '';
    });

  
function displayUserPhoto(photo) {
  const chatWindow = document.getElementById('chat-window');
  const photoBubble = document.createElement('div');
  photoBubble.classList.add('pat-bubble');
  
  const img = document.createElement('img');
  img.src = URL.createObjectURL(photo);
  img.alt = 'User Photo';

  // Resize the image to 200x200 pixels
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 200;
  canvas.height = 200;
  
  // Ensure the image is loaded before drawing on the canvas
  img.onload = function() {
    ctx.drawImage(img, 0, 0, 200, 200);

    // Create a new image with the resized data URL
    const resizedImg = new Image();
    resizedImg.src = canvas.toDataURL('image/png');

    // Add the resized image to the chat window
    photoBubble.appendChild(resizedImg);
    chatWindow.appendChild(photoBubble);

  };
}



  function getBotResponse(userMessage) {
  const url = '/analyse';

  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symptome: userMessage })
  })
    .then(response => response.json())
    .then(data => {
      const { response } = data;
      // Print the user message
      const userBubble = document.createElement('div');
      userBubble.classList.add('pat-bubble');
      userBubble.textContent = userMessage;
      chatWindow.appendChild(userBubble);

      // Print the answer from our NLP
      const replyBubble = document.createElement('div');
      replyBubble.classList.add('pat-bubble', 'bot-bubble');
      replyBubble.textContent = response;
      chatWindow.appendChild(replyBubble);
    });
}



function resnetResponse(uploadedPhoto) {
  const url = '/transfor'; 

  const formData = new FormData();
  formData.append('photo', uploadedPhoto);

  fetch(url, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      const { response } = data;

      // Print the answer from  ResNet 50 model
      const replyBubble = document.createElement('div');
      replyBubble.classList.add('pat-bubble', 'bot-bubble');
      replyBubble.textContent = response;
      chatWindow.appendChild(replyBubble);
    });
}


function transResponse(uploadedPhoto) {
  const url = '/transfor'; 

  const formData = new FormData();
  formData.append('photo', uploadedPhoto);

  fetch(url, {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      const { response } = data;
    
      // Print the answer from Vision-Transformer model
      const replyBubble = document.createElement('div');
      replyBubble.classList.add('pat-bubble', 'bot-bubble');
      replyBubble.textContent = response;
      chatWindow.appendChild(replyBubble);
    });
}

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import {getFirestore} from "firebase/firestore"
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyA4_8ry7bsJx8FxltCiJ5dJvSADeBufMZk",
  authDomain: "ai-travel-planner-app-e0124.firebaseapp.com",
  projectId: "ai-travel-planner-app-e0124",
  storageBucket: "ai-travel-planner-app-e0124.firebasestorage.app",
  messagingSenderId: "801471858748",
  appId: "1:801471858748:web:c3eeaf7e34deedd4995b39"
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
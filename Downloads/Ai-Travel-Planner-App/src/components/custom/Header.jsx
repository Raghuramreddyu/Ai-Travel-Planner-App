import React from 'react'
import { useState } from 'react'
import { Button } from '../ui/button'
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
  } from "@/components/ui/popover"
import { googleLogout } from '@react-oauth/google'
import { useGoogleLogin } from '@react-oauth/google'
import axios from 'axios'
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
  } from "@/components/ui/dialog";
  
const Header = () => {
    const user = JSON.parse(localStorage.getItem('user'))
     const [openDialog, setOpenDialog] = useState(false);
     const login = useGoogleLogin({
        onSuccess: (tokenResp) => GetUserProfile(tokenResp),
        onError: (error) => console.log(error),
      });
      const GetUserProfile = (tokenInfo) => {
        axios
          .get(
            `https://www.googleapis.com/oauth2/v1/userinfo?access_token=${tokenInfo?.access_token}`,
            {
              headers: {
                Authorization: `Bearer ${tokenInfo?.access_token}`,
                Accept: "application/json",
              },
            }
          )
          .then((resp) => {
            localStorage.setItem("user", JSON.stringify(resp.data));
            setOpenDialog(false);
            window.location.reload()
         
          })
          .catch((error) => {
            console.error("Error fetching user profile:", error);
          });
      };

    return (
        <div className="max-w-screen-3xl mx-auto px-4 mt-3 ">
        <div className="flex flex-wrap justify-between items-center shadow-xl h-auto min-h-[96px] px-4 sm:px-10 py-4 rounded-lg">
            <img src="/logo.png" alt="logo" className="h-[60px] w-[60px] sm:h-[70px] sm:w-[70px] rounded-lg" />
            <div>
                {user ?
                <div className='flex items-center gap-3'>
                    <a href="/mytrips">
                    <Button variant='outline' className='rounded-full'>My trips</Button>
                    </a>
                    <Popover>
                <PopoverTrigger> <img src={user?.picture} className='h-[35px] w-[35px] rounded-full' alt="" />
                </PopoverTrigger>
               <PopoverContent
               ><h2 className='cursor-pointer'
               onClick={()=>{
                googleLogout;
                localStorage.clear()
                window.location.reload()
               }}
               >Logout</h2>
               </PopoverContent>
                </Popover>
                   
                </div>: <Button onClick={()=>setOpenDialog(true)} className="mt-3 sm:mt-0">SignIn!</Button>}
     <Dialog open={openDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogDescription>
              <img src="/logo.png" alt="Logo" />
              <h2 className="font-bold text-lg mt-7">Sign in with Google</h2>
              <p>Sign in to the app securely using Google authentication.</p>
              <Button
                onClick={login}
                className="w-full mt-5 flex gap-4 items-center"
              >
                Sign in with Google
              </Button>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
            </div>
           
        </div>
    </div>
    )
}

export default Header

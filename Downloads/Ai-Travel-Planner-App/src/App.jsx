import { useState } from 'react'
import { Button } from './components/ui/button'
import Header from './components/custom/Header'
import Hero from './components/custom/Hero'
function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        {/* <Header/> */}
        <Hero/>
      </div>
    </>
  )
}

export default App

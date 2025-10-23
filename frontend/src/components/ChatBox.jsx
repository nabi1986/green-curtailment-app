import { useEffect, useRef, useState } from 'react'

export default function ChatBox() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const wsRef = useRef(null)

  useEffect(() => {
    const url = `${import.meta.env.VITE_WS_BASE}/ws/chat`
    const ws = new WebSocket(url)
    wsRef.current = ws
    ws.onmessage = (evt) => {
      setMessages(m => [...m, { from:"bot", text: evt.data }])
    }
    ws.onopen = () => setMessages(m => [...m, { from:"sys", text:"Connected." }])
    ws.onclose = () => setMessages(m => [...m, { from:"sys", text:"Disconnected." }])
    return () => ws.close()
  }, [])

  const send = () => {
    const txt = input.trim()
    if (!txt || !wsRef.current) return
    wsRef.current.send(txt)
    setMessages(m => [...m, { from:"me", text: txt }])
    setInput("")
  }

  return (
    <div style={{border:'1px solid #ddd', borderRadius:8, padding:8, height:320, display:'flex', flexDirection:'column', gap:8}}>
      <div style={{flex:1, overflowY:'auto', padding:4}}>
        {messages.map((m,i)=>(
          <div key={i} style={{margin:'6px 0', textAlign: m.from==="me"?'right':'left', opacity: m.from==="sys"?0.6:1}}>
            <span style={{display:'inline-block', padding:'6px 10px', borderRadius:8, background: m.from==="me"?'#eef':'#f5f5f5'}}>{m.text}</span>
          </div>
        ))}
      </div>
      <div style={{display:'flex', gap:6}}>
        <input
 value={input}
  onChange={e=>setInput(e.target.value)}
  placeholder="Type: what's the curtailment probability?"
  style={{flex:1, padding:8, border:'1px solid #ccc', borderRadius:6}}
/>
<button onClick={send} style={{padding:'8px 14px', borderRadius:6, border:'1px solid #ccc', cursor:'pointer'}}>
  Send
</button>
      </div>
    </div>
  )
}

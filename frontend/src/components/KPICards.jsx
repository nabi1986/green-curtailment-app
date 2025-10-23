export default function KPICards({ latest }) {
  if (!latest) return null
  return (
    <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:'12px',margin:'12px 0'}}>
      <div style={{border:'1px solid #ddd',borderRadius:8,padding:12}}>
        <div>Predicted Generation (MW)</div>
        <div style={{fontSize:24,fontWeight:700}}>{latest.gen_mw}</div>
      </div>
      <div style={{border:'1px solid #ddd',borderRadius:8,padding:12}}>
        <div>Price (€/MWh)</div>
        <div style={{fontSize:24,fontWeight:700}}>{latest.price_eur_mwh}</div>
      </div>
      <div style={{border:'1px solid #ddd',borderRadius:8,padding:12}}>
        <div>Curtailment Probability</div>
        <div style={{fontSize:24,fontWeight:700}}>{Math.round(latest.curtailment_prob*100)}%</div>
      </div>
    </div>
  )
}

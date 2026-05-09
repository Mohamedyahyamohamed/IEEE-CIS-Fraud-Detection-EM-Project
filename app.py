import streamlit as st
import pandas as pd
import pickle

# 1. إعدادات الصفحة
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

# 2. تحميل الموديل وأسماء الأعمدة (مع استخدام الـ Cache لتسريع التطبيق)
@st.cache_resource
def load_system():
    # تحميل الموديل
    with open('fraud_pipeline.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    # تحميل أسماء الأعمدة اللي الموديل اتدرب عليها
    with open('model_columns.pkl', 'rb') as file:
        loaded_columns = pickle.load(file)
        
    return loaded_model, loaded_columns

# استدعاء الدالة
model, expected_columns = load_system()

# 3. واجهة المستخدم (الـ Header)
st.title("🛡️ نظام الفحص الفوري للعمليات")
st.markdown("قم بإدخال بيانات العملية الحالية للتحقق من احتمالية وجود احتيال (Fraud).")
st.divider()

# 4. بناء الـ Form لإدخال البيانات
with st.form("transaction_form"):
    st.subheader("📝 بيانات العملية")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transaction_amt = st.number_input("مبلغ العملية ($ TransactionAmt)", min_value=0.0, value=150.0)
        # يمكنك إضافة حقول أخرى هنا مستقبلاً
        
    with col2:
        addr1 = st.number_input("الرقم البريدي / المنطقة (addr1)", min_value=0.0, value=315.0)
        # يمكنك إضافة حقول أخرى هنا مستقبلاً
        
    with col3:
        st.info("💡 سيتم تعويض باقي البيانات بأرقام افتراضية لاختبار عمل الموديل.")

    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔍 فحص العملية الآن", use_container_width=True)

# 5. منطق الفحص (الـ Logic) عند الضغط على الزر
if submit_button:
    with st.spinner("جاري تحليل بصمة العملية المذكورة واستخراج نسبة الخطر..."):
        
        # إنشاء قاموس مليء بالأصفار لكل الأعمدة المطلوبة
        input_dict = {col: 0 for col in expected_columns}
        
        # تحديث القيم الحقيقية التي أدخلها المستخدم
        if 'TransactionAmt' in input_dict: 
            input_dict['TransactionAmt'] = transaction_amt
        if 'addr1' in input_dict: 
            input_dict['addr1'] = addr1
            
        # تحويل القاموس إلى DataFrame
        input_df = pd.DataFrame([input_dict])
        
        try:
            # التوقع المباشر (0 أو 1)
            prediction = model.predict(input_df)
            
            # حساب نسبة الشك (الـ Probability)
            probabilities = model.predict_proba(input_df)[0]
            fraud_probability = probabilities[1] * 100 # نسبة تصنيفها كـ 1
            
            st.divider()
            
            # عرض النسبة في شريط تقدم (Progress Bar)
            st.write(f"📊 **نسبة احتمالية الاحتيال:** %{fraud_probability:.2f}")
            st.progress(int(fraud_probability) if fraud_probability <= 100 else 100)
            
            # اتخاذ القرار بناءً على النسبة (يمكنك تعديل الـ 50 دي حسب صرامة النظام عندك)
            if prediction[0] == 1 or fraud_probability >= 50.0:
                st.error("🚨 **تحذير أمني:** هذه العملية تم تصنيفها كعملية احتيال محتملة (FRAUD)! الرجاء إيقاف التنفيذ ومراجعة بيانات العميل.")
            else:
                st.success("✅ **عملية آمنة:** بصمة هذه العملية طبيعية ولا يوجد بها نشاط مشبوه.")
                if fraud_probability < 10:
                    st.balloons() # احتفال لو العملية آمنة جداً
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء الفحص. التفاصيل التقنية: {e}")
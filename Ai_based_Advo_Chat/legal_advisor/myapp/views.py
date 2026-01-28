from django.shortcuts import render, redirect, HttpResponse, HttpResponseRedirect
from .models import *
from django.db.models import Q
from django.contrib.auth import authenticate, logout, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from datetime import datetime
import datetime as dt

import nltk

nltk.download("punkt")
nltk.download("wordnet")


def index(request):
    return render(request, "index.html")


def contact(request):
    return render(request, "contact.html")


# Create your views here.


def logins(request):
    if request.POST:
        email = request.POST["email"]
        passw = request.POST["password"]
        print(email)
        print(passw)
        user = authenticate(username=email, password=passw)
        print(user)

        if user is not None:
            login(request, user)
            if user.userType == "Admin":
                messages.info(request, "Login Success")
                return redirect("/admhome")
            elif user.userType == "User":
                id = user.id
                email = user.username
                request.session["uid"] = id
                request.session["email"] = email
                messages.info(request, "Login Success")
                return redirect("/user_home")
            elif user.userType == "Advocate":
                id = user.id
                email = user.username
                request.session["aid"] = id
                request.session["email"] = email

                messages.info(request, "Login Success")
                return redirect("/advocatehome")
        else:
            print("Hiii")
            messages.error(request, "Invalid Username/Password")
    return render(request, "login.html")


def Userregister(request):
    current_date = datetime.today().strftime("%Y-%m-%d")
    print(current_date)
    if request.POST:
        name = request.POST["name"]
        email = request.POST["email"]
        phone = request.POST["phone"]
        dob = request.POST["dob"]
        gender = request.POST["gender"]
        address = request.POST["address"]
        ac_no = request.POST["ac_no"]
        aadhaar = request.POST["aadhaar"]
        password = request.POST["password"]
        image = request.FILES["image"]

        if Login.objects.filter(username=email).exists():
            messages.error(request, "Email is Already Exists")
            return redirect("/login")
        else:
            logUser = Login.objects.create_user(
                username=email,
                password=password,
                userType="User",
                viewPass=password,
                is_active=1,
            )
            logUser.save()

            userReg = UserRegistration.objects.create(
                name=name,
                email=email,
                phone=phone,
                gender=gender,
                address=address,
                Account_no=ac_no,
                aadhaar=aadhaar,
                dob=dob,
                loginid=logUser,
                image=image,
            )
            userReg.save()
    return render(request, "userReg.html", {"current_date": current_date})


def AdvocateReg(request):
    view=Case_Category.objects.all()
    current_date = datetime.today().strftime("%Y-%m-%d")
    print(current_date)
    if request.POST:
        name = request.POST["name"]
        email = request.POST["email"]
        phone = request.POST["phone"]
        dob = request.POST["dob"]
        gender = request.POST["gender"]
        address = request.POST["address"]
        password = request.POST["password"]
        district = request.POST["district"]
        education = request.POST["education"]
        category = request.POST["cases"]
        image = request.FILES["image"]
        files = request.FILES["files"]

        if Login.objects.filter(username=email).exists():
            messages.error(request, "Email is Already Exists")
            return redirect("/login")
        else:
            logUser = Login.objects.create_user(
                username=email,
                password=password,
                userType="Advocate",
                viewPass=password,
                is_active=0,
            )
            logUser.save()

            userReg = Advocate.objects.create(
                name=name,
                email=email,
                phone=phone,
                files=files,
                gender=gender,
                address=address,
                dob=dob,
                category=category,
                qualification=education,
                district=district,
                loginid=logUser,
                image=image,
            )
            userReg.save()
            return HttpResponseRedirect("/Regpay?email=" + email)
    return render(request, "AdvocateReg.html", {"current_date": current_date,"view": view})


def Regpay(request):
    emial = request.GET.get("email")
    if request.POST:
        up = Advocate.objects.filter(email=emial).update(Regfee="Paid")
        messages.info(request, "Registered Successfully")
        return redirect("/login")
    return render(request, "Regpayment.html")


def admhome(request):
    return render(request, "Admins/index.html")


def advocatehome(request):
    return render(request, "Advocate/index.html")


def userhome(request):
    return render(request, "User/index.html")


def admviewusers(request):
    view = UserRegistration.objects.all()
    return render(request, "Admins/users.html", {"view": view})


def admviewadvocate(request):
    view = Advocate.objects.all()
    return render(request, "Admins/advocates.html", {"view": view})

def ipc_sections(request):
    view = IPC_sections.objects.all()
    return render(request, "Admins/ipc_sections.html", {"view": view})

def add_case_category(request):
    if request.POST:
        Category = request.POST.get("Category")
        Decsription = request.POST.get("desc")

        add = Case_Category.objects.create(Category=Category, Description=Decsription)
        add.save()
        messages.info(request, "Category Added Successfully")
    return render(request, "Admins/add_case_category.html")


def addipc_section(request):
    if request.POST:
        ipc = request.POST.get("ipc")
        Decsription = request.POST.get("desc")

        add = IPC_sections.objects.create(IPC=ipc, Description=Decsription)
        add.save()
        messages.info(request, "IPC Sections Added Successfully")
    return render(request, "Admins/addipc_section.html")


def userview_Advocate(request):
    vie = Advocate.objects.filter(loginid__is_active=1)
    print(vie)
    return render(request, "User/advocate_view.html", {"vie": vie})


def user_home(request):
    return render(request, "User/index.html")


def user_add_feed(request):
    uid = request.session["uid"]
    Uid = UserRegistration.objects.get(loginid=uid)
    current_date = datetime.today().strftime("%Y-%m-%d")
    if request.POST:
        sub = request.POST.get("exampleForm")
        feedback = request.POST.get("feedback")

        add = Feedback.objects.create(
            sub=sub, Feedback=feedback, uid=Uid, date=current_date, Type="User"
        )
        add.save()
        messages.info(request, "Feedback Successfully")
    return render(request, "User/add_feedback.html")


def user_view_ipc(request):
    view = IPC_sections.objects.all()
    return render(request, "User/view_ipc.html", {"view": view})


def user_book_case(request):
    email = request.GET.get("email")
    aid = Advocate.objects.get(email=email)
    uid = request.session["uid"]
    Uid = UserRegistration.objects.get(loginid=uid)
    current_date = datetime.today().strftime("%Y-%m-%d")
    view = Advocate.objects.filter(email=email)
    if request.POST:
        sub = request.POST.get("sub")
        desc = request.POST.get("desc")

        add = Case_request.objects.create(
            user=Uid, advocate=aid, sub=sub, date=current_date, desc=desc
        )
        add.save()
        messages.info(request, "Request Added Successfully")
    return render(request, "User/case_book.html", {"view": view})


def user_view_request(request):
    uid = request.session["uid"]
    email = request.session["email"]
    Uid = UserRegistration.objects.get(loginid=uid)
    view = Case_request.objects.filter(user__email=email)
    return render(request, "User/user_view_request.html", {"view": view})


def adv_view_request(request):
    uid = request.session["aid"]
    Uid = Advocate.objects.get(loginid=uid)

    print("Advocate :", uid)
    view = Case_request.objects.filter(request="pending", advocate=Uid.id)
    ap = Case_request.objects.filter(request="Approved", advocate=Uid.id)
    rj = Case_request.objects.filter(request="Rejected", advocate=Uid.id)
    return render(
        request, "Advocate/view_case_request.html", {"view": view, "ap": ap, "rj": rj}
    )


def approve_case_request(request):
    rid = request.GET.get("id")
    approve = Case_request.objects.filter(id=rid).update(request="Approved")
    messages.info(request, "Request Accepted")
    return redirect("/adv_view_request")


def reject_case_request(request):
    rid = request.GET.get("id")
    approve = Case_request.objects.filter(id=rid).update(request="Rejected")
    messages.info(request, "Request Rejected")
    return redirect("/adm_view_request")


def approve_advocate(request):
    id = request.GET.get("id")
    approve = Login.objects.filter(id=id).update(is_active=1)
    messages.info(request, "Request Accepted")
    return redirect("/admviewadvocate")


def reject_advocate(request):
    id = request.GET.get("id")
    reject = Login.objects.filter(id=id).update(is_active=0)
    messages.info(request, "Request Rejected")
    return redirect("/admviewadvocate")


def advocate_add_feed(request):
    aid = request.session["aid"]
    aaid = Advocate.objects.get(loginid=aid)
    current_date = datetime.today().strftime("%Y-%m-%d")
    if request.POST:
        sub = request.POST.get("exampleForm")
        feedback = request.POST.get("feedback")

        add = Feedback.objects.create(
            sub=sub, Feedback=feedback, aid=aaid, date=current_date, Type="Advocate"
        )
        add.save()
        messages.info(request, "Feedback Added Successfully")
    return render(request, "Advocate/add_feedback.html")


def adv_view_ipc(request):
    view = IPC_sections.objects.all()
    return render(request, "Advocate/view_ipc.html", {"view": view})


def adv_view_approved_case(request):
    current_date = datetime.today().strftime("%Y-%m-%d")
    print(current_date)
    aid = request.session["aid"]
    aaid = Advocate.objects.get(loginid=aid)
    rid = request.GET.get("id")
    rrid = Case_request.objects.get(id=rid)
    uid = request.GET.get("uid")
    uuid = UserRegistration.objects.get(id=uid)
    print("userid", uuid.id)
    view = Case_request.objects.filter(request="Approved", id=rid)

    if "files" in request.POST:
        status = request.POST.get("status")
        date = request.POST.get("date")
        case = Case_request.objects.filter(id=rid).update(date=date, status=status)
        return HttpResponse(
            "<script>alert('Status Updated');window.location='/adv_view_request'</script>"
        )

    elif "Describe" in request.POST:
        desc = request.POST.get("desc")
        fees = request.POST.get("fees")
        file = request.FILES["file"]
        if Case_details.objects.filter(desc=desc, fees=fees).exists():
            messages.success(request, "Already Exists")
        else:
            upload = Case_details.objects.create(
                desc=desc,
                fees=fees,
                rid_id=rrid.id,
                file=file,
                user_id=uuid.id,
                advocate_id=aaid.id,
            )
            upload.save()
            messages.success(request, "Files Uploaded")
            return HttpResponse(
                "<script>alert('File And Fees Uploaded');window.location='/adv_view_request'</script>"
            )
        # else:
        #     messages.info(request, "Please Select Petitions from Case Request")
    return render(
        request, "Advocate/cases.html", {"view": view, "current_date": current_date}
    )


def user_profile(request):
    uid = request.session["uid"]
    view = UserRegistration.objects.filter(loginid=uid)

    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        address = request.POST.get("address")
        password = request.POST.get("password")
        image = request.FILES.get("image")

        if image:
            userid = UserRegistration.objects.get(loginid=uid)
            userid.image = image
            userid.save()

            up = UserRegistration.objects.filter(loginid=uid).update(
                name=name, email=email, phone=phone, address=address
            )

            pas_up = Login.objects.get(id=uid)
            pas_up.set_password(password)
            pas_up.viewPass = password
            pas_up.save()

            HttpResponse(
                "<script>alert('profile updated');window.location='/user_profile'</script>"
            )

        else:
            up = UserRegistration.objects.filter(loginid=uid).update(
                name=name, email=email, phone=phone, address=address
            )

            pas_up = Login.objects.get(id=uid)
            pas_up.set_password(password)
            pas_up.viewPass = password
            pas_up.save()

            HttpResponse(
                "<script>alert('profile updated');window.location='/user_profile'</script>"
            )

    return render(request, "User/profile.html", {"view": view})


def user_case_files(request):
    rid = request.GET.get("rid")
    uid = request.session["uid"]
    Uid = UserRegistration.objects.get(loginid=uid)
    view = Case_details.objects.filter(user=Uid, rid=rid)
    return render(request, "User/user_case_files.html", {"view": view})


def chat(request):
    import chatgui

    return redirect("/")


def payment_page(request):
    amut = request.GET.get("amt")
    cid = request.GET.get("cid")

    if request.POST:
        up = Case_details.objects.filter(id=cid, status="Not_Paid").update(
            status="Paid"
        )
        return redirect("/Applied_success")
    return render(request, "User/payment_page.html", {"amount": amut})


def payment_view(request):
    uid = request.session["uid"]
    view = Case_details.objects.filter(user=uid)
    return render(request, "User/payment_details.html", {"view": view})


def Applied_success(request):
    return render(request, "User/Applied_success.html")


def userchat(request):
    uid = request.session["uid"]
    Uid = UserRegistration.objects.get(loginid=uid)
    # Artists
    name = ""
    pimage = ""
    aid = request.GET.get("aid")
    email = request.GET.get("email")
    artistData = Advocate.objects.filter(email=email)
    getChatData = Chat.objects.filter(Q(uid=Uid) & Q(Advo__email=email))
    print("getChatData ", getChatData)
    current_time = datetime.now().time()
    formatted_time = current_time.strftime("%H:%M")
    userid = UserRegistration.objects.get(loginid__id=uid)
    if aid:
        advocate_id = Advocate.objects.get(email=email)
        name = advocate_id.name
        pimage = advocate_id.image
    if request.POST:
        message = request.POST["message"]
        sendMsg = Chat.objects.create(
            uid=userid,
            message=message,
            Advo=advocate_id,
            time=formatted_time,
            utype="USER",
        )
        sendMsg.save()
    return render(
        request,
        "User/chat.html",
        {
            "artistData": artistData,
            "getChatData": getChatData,
            "artistid": name,
            "image": pimage,
        },
    )


def reply(request):
    email = request.session["email"]
    uid = request.session["aid"]
    Aid = Advocate.objects.get(loginid=uid)
    print(uid)
    name = ""
    pimage = ""
    id = request.GET.get("uid")
    uemail = request.GET.get("email")
    userData = UserRegistration.objects.filter(email=uemail)
    print("User Details :", id)
    UserId = UserRegistration.objects.get(email=uemail)
    getChatData = Chat.objects.filter(Q(Advo__loginid=uid) & Q(uid__email=uemail))
    print("HELLO", getChatData)
    current_time = datetime.now().time()
    formatted_time = current_time.strftime("%H:%M")
    advo_id = Advocate.objects.get(email=email)
    advo = advo_id.id
    if id:
        userid = UserRegistration.objects.get(loginid=id)
        name = userid.name
        pimage = userid.image
        print("userid : ", userid)
        print("name : ", name)
        print("image : ", pimage)
    if request.POST:
        message = request.POST["message"]
        sendMsg = Chat.objects.create(
            uid=userid,
            message=message,
            Advo_id=advo,
            time=formatted_time,
            utype="ADVOCATE",
        )
        sendMsg.save()
    return render(
        request,
        "Advocate/chat.html",
        {
            "userData": userData,
            "getChatData": getChatData,
            "userid": name,
            "image": pimage,
        },
    )


def admviewfeedback(request):
    view = Feedback.objects.all()
    return render(request, "Admins/view_feedback.html", {"view": view})


def udp(request):
    dele = Login.objects.filter(id=9).update(is_active=1)
    return HttpResponse("deleted")


def delete_feedback(request):
    fid = request.GET.get("fid")
    if fid:
        dele = Feedback.objects.filter(id=fid).delete()
        return HttpResponse(
            "<script>alert('Feedback Deleted');window.location='/admviewfeedback';</script>"
        )


def add_rating(request):
    uid = request.session.get("uid")
    Uid = UserRegistration.objects.get(loginid=uid)
    aid = request.GET.get("aid")
    email = request.GET.get("email")
    Aid = Advocate.objects.get(loginid=aid)
    rid = request.GET.get("rid")
    Rid = Case_request.objects.get(id=rid)

    if request.method == "POST":
        rating = request.POST.get("rating")
        if Rating.objects.filter(uid=Uid,aid=Aid,rid=Rid).exists():
            return HttpResponse(
                "<script>alert('Rating Already Added');window.location='/user_view_request';</script>"
            )
        else:
            add = Rating.objects.create(aid=Aid, rid=Rid, rating=rating, uid=Uid)
            add.save()
            ratings = Rating.objects.filter(aid=Aid)
            total_ratings = ratings.count()
            if total_ratings > 0:
                total_sum = sum(r.rating for r in ratings)
                avg_rating = total_sum / total_ratings
                print("totalSum: ", total_sum)
                print("Avg.rating: ", avg_rating)
                update = Advocate.objects.filter(email=email).update(
                    advo_rating=avg_rating
                )
                return HttpResponse(
                    "<script>alert('Rating Added');window.location='/user_view_request';</script>"
                )

    return render(request, "User/add_rating.html")
